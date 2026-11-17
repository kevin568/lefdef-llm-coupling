"""Export routed geometry through OpenDB: per net wire rectangles, via count, pins, centreline wire length.
Usage: openroad -no_init -no_splash -exit -python export_geo.py <lef1,lef2,...> <def> <out.json>"""
import json
import sys
from pathlib import Path
import odb

lefs, deff, out = sys.argv[1:4]
db = odb.dbDatabase.create()
for l in lefs.split(','):
    odb.read_lef(db, l)
odb.read_def(db.getTech(), deff)
block = db.getChip().getBlock(); tech = db.getTech()
layers = {l.getName(): dict(level=l.getRoutingLevel(), pitch=l.getPitch(), width=l.getWidth(), direction=str(l.getDirection()))
          for l in tech.getLayers() if l.getRoutingLevel()}
fallback = None
if not hasattr(odb, 'dbWirePathItr'):   # older OpenROAD builds: read the wire shapes from the DEF text instead
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib'))
    import defwires

    def width_of(name):
        l = tech.findLayer(name)
        return l.getWidth() if l and l.getRoutingLevel() else 0

    def via_layers(name):
        v = block.findVia(name) or tech.findVia(name)
        if not v:
            return set()
        return {b.getTechLayer().getName() for b in v.getBoxes() if b.getTechLayer() and b.getTechLayer().getRoutingLevel()}
    fallback = defwires.net_wires(deff, width_of, via_layers)
    fallback.update({k.replace('\\', ''): v for k, v in list(fallback.items())})
nets = {}
for n in block.getNets():
    if n.getSigType() in ('POWER', 'GROUND'):
        continue
    wire = n.getWire(); shapes = []; vias = 0
    if wire and fallback is not None:
        fs, vias = fallback.get(n.getName(), fallback.get(n.getName().replace('\\', ''), ([], 0)))
        shapes = [dict(layer=l, rect=list(r)) for l, r in fs]
    elif wire:
        it = odb.dbWirePathItr(); it.begin(wire)
        path = odb.dbWirePath(); ps = odb.dbWirePathShape()
        while it.getNextPath(path):
            while it.getNextShape(ps):
                shape = ps.shape
                if shape.isVia():
                    vias += 1; continue
                b = shape.getBox()
                shapes.append(dict(layer=shape.getTechLayer().getName(), rect=[b.xMin(), b.yMin(), b.xMax(), b.yMax()]))
    pins = []
    for p in n.getITerms():
        ok, x, y = p.getAvgXY()
        if not ok:
            b = p.getBBox(); x, y = (b.xMin() + b.xMax()) // 2, (b.yMin() + b.yMax()) // 2
        pins.append(dict(name=p.getInst().getName() + '/' + p.getMTerm().getName(), xy=[x, y], macro=p.getInst().getMaster().isBlock()))
    for p in n.getBTerms():
        boxes = [b for bp in p.getBPins() for b in bp.getBoxes()]
        if not boxes:
            continue
        b = boxes[0]
        pins.append(dict(name='PORT/' + p.getName(), xy=[(b.xMin() + b.xMax()) // 2, (b.yMin() + b.yMax()) // 2], macro=False))
    # centreline length: OpenDB rectangles carry width/2 end extensions
    wl = sum(max(s['rect'][2] - s['rect'][0], s['rect'][3] - s['rect'][1]) - min(s['rect'][2] - s['rect'][0], s['rect'][3] - s['rect'][1]) for s in shapes)
    nets[n.getName()] = dict(shapes=shapes, pins=pins, vias=vias, wl_dbu=wl, routed=bool(wire), signal_type=str(n.getSigType()))
die = block.getDieArea()
Path(out).write_text(json.dumps(dict(design=block.getName(), dbu=block.getDbUnitsPerMicron(), layers=layers, nets=nets,
                                     die=[die.xMin(), die.yMin(), die.xMax(), die.yMax()], instances=len(block.getInsts()))) + '\n')
print('FLOW_DONE', len(nets), 'nets', flush=True)
