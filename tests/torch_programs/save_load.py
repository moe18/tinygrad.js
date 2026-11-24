# torch.save / torch.load: round trips, and reading a checkpoint that real PyTorch wrote.
import base64, io, os, tempfile, copy
from collections import OrderedDict
out = {}
# written by real PyTorch 2.x with torch.save (zip format); see tests/reference/generate_torch.py
REAL_PYTORCH_CHECKPOINT = (
  "UEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAQABIAYXJjaGl2ZS9kYXRhLnBrbEZCDgBaWlpaWlpaWlpaWlpaWoACfXEAKFgFAAAAbW9kZWxxAWNjb2xsZWN0aW9ucwpPcmRlcmVkRGljdApxAilScQ"
  "MoWAgAAAAwLndlaWdodHEEY3RvcmNoLl91dGlscwpfcmVidWlsZF90ZW5zb3JfdjIKcQUoKFgHAAAAc3RvcmFnZXEGY3RvcmNoCkZsb2F0U3RvcmFnZQpxB1gBAAAAMHEIWAMAAABjcHVxCUsGdHEK"
  "UUsASwJLA4ZxC0sDSwGGcQyJaAIpUnENdHEOUnEPWAYAAAAwLmJpYXNxEGgFKChoBmgHWAEAAAAxcRFoCUsCdHESUUsASwKFcRNLAYVxFIloAilScRV0cRZScRdYCAAAADEud2VpZ2h0cRhoBSgoaA"
  "ZoB1gBAAAAMnEZaAlLAnRxGlFLAEsChXEbSwGFcRyJaAIpUnEddHEeUnEfWAYAAAAxLmJpYXNxIGgFKChoBmgHWAEAAAAzcSFoCUsCdHEiUUsASwKFcSNLAYVxJIloAilScSV0cSZScSdYDgAAADEu"
  "cnVubmluZ19tZWFucShoBSgoaAZoB1gBAAAANHEpaAlLAnRxKlFLAEsChXErSwGFcSyJaAIpUnEtdHEuUnEvWA0AAAAxLnJ1bm5pbmdfdmFycTBoBSgoaAZoB1gBAAAANXExaAlLAnRxMlFLAEsChX"
  "EzSwGFcTSJaAIpUnE1dHE2UnE3WBUAAAAxLm51bV9iYXRjaGVzX3RyYWNrZWRxOGgFKChoBmN0b3JjaApMb25nU3RvcmFnZQpxOVgBAAAANnE6aAlLAXRxO1FLACkpiWgCKVJxPHRxPVJxPnV9cT9Y"
  "CQAAAF9tZXRhZGF0YXFAaAIpUnFBKFgAAAAAcUJ9cUNYBwAAAHZlcnNpb25xREsBc1gBAAAAMHFFfXFGaERLAXNYAQAAADFxR31xSGhESwJzdXNiWAUAAABlcG9jaHFJSwNYBAAAAGJlc3RxSkc/wA"
  "AAAAAAAFgEAAAAdGFnc3FLXXFMKFgBAAAAYXFNWAEAAABicU5lWAMAAABpZHNxT2gFKChoBmg5WAEAAAA3cVBoCUsEdHFRUUsASwSFcVJLAYVxU4loAilScVR0cVVScVZYBAAAAGhhbGZxV2gFKCho"
  "BmN0b3JjaApEb3VibGVTdG9yYWdlCnFYWAEAAAA4cVloCUsEdHFaUUsBSwKFcVtLAoVxXIloAilScV10cV5ScV9YBgAAAG5lc3RlZHFgfXFhKFgBAAAAdHFiaAUoKGgGY3RvcmNoCkludFN0b3JhZ2"
  "UKcWNYAQAAADlxZGgJSwR0cWVRSwBLAksChnFmSwFLAoZxZ4loAilScWh0cWlScWpYBAAAAGZsYWdxa4hYBAAAAG5vbmVxbE5YBQAAAHR1cGxlcW1LAUdABAAAAAAAAIZxbnV1LlBLBwh3hUs5GgQA"
  "ABoEAABQSwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAABcAIQBhcmNoaXZlLy5mb3JtYXRfdmVyc2lvbkZCHQBaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWjFQSwcIt+/cgwEAAAABAAAAUEsDBA"
  "AACAgAAAAAAAAAAAAAAAAAAAAAAAAaADcAYXJjaGl2ZS8uc3RvcmFnZV9hbGlnbm1lbnRGQjMAWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaNjRQSwcI"
  "P3dx6QIAAAACAAAAUEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAARAD8AYXJjaGl2ZS9ieXRlb3JkZXJGQjsAWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWl"
  "paWlpaWlpaWlpsaXR0bGVQSwcIhT3jGQYAAAAGAAAAUEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAOAD4AYXJjaGl2ZS9kYXRhLzBGQjoAWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa"
  "WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWgAAAD8AAIC/AAAAQAAAwD8AAIA+AABAv1BLBwgvpNNHGAAAABgAAABQSwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAAA4ALABhcmNoaXZlL2RhdGEvMUZCKA"
  "BaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpazczMPc3MTL5QSwcI9R0g3AgAAAAIAAAAUEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAOADwAYXJjaGl2ZS9kYXRhLzJGQjgA"
  "WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWloAAIA/AACAP1BLBwiQUQbVCAAAAAgAAABQSwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAAA4APABhcm"
  "NoaXZlL2RhdGEvM0ZCOABaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWgAAAAAAAAAAUEsHCGnfImUIAAAACAAAAFBLAwQAAAgIAAAAAAAAAAAA"
  "AAAAAAAAAAAADgA8AGFyY2hpdmUvZGF0YS80RkI4AFpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpampmZPs3MzD5QSwcIxdn2PwgAAAAIAAAAUE"
  "sDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAOADwAYXJjaGl2ZS9kYXRhLzVGQjgAWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWloAAIA/AACAP1BL"
  "BwiQUQbVCAAAAAgAAABQSwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAAA4APABhcmNoaXZlL2RhdGEvNkZCOABaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWl"
  "paWlpaWgcAAAAAAAAAUEsHCHDW528IAAAACAAAAFBLAwQAAAgIAAAAAAAAAAAAAAAAAAAAAAAADgA8AGFyY2hpdmUvZGF0YS83RkI4AFpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa"
  "WlpaWlpaWlpaWlpaWlpaWlpaWlpaAAAAAAAAAAABAAAAAAAAAAIAAAAAAAAAAwAAAAAAAABQSwcIChIAkSAAAAAgAAAAUEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAOACQAYXJjaGl2ZS9kYXRhLz"
  "hGQiAAWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWloAAAAAAADwPwAAAAAAAPA/AAAAAAAA8D8AAAAAAADwP1BLBwhssp17IAAAACAAAABQSwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAAA4A"
  "JABhcmNoaXZlL2RhdGEvOUZCIABaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWgEAAAACAAAAAwAAAAQAAABQSwcI79QFrxAAAAAQAAAAUEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAPAD"
  "MAYXJjaGl2ZS92ZXJzaW9uRkIvAFpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaMwpQSwcI0Z5nVQIAAAACAAAAUEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAe"
  "ADIAYXJjaGl2ZS8uZGF0YS9zZXJpYWxpemF0aW9uX2lkRkIuAFpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWloxMjM3NzYxMDQwODY5NDQ1ODA2NzAyMTEzNTI4Mz"
  "g3MjgyMzQ1MTA0UEsHCA6/I9soAAAAKAAAAFBLAQIAAAAACAgAAAAAAAB3hUs5GgQAABoEAAAQAAAAAAAAAAAAAAAAAAAAAABhcmNoaXZlL2RhdGEucGtsUEsBAgAAAAAICAAAAAAAALfv3IMBAAAA"
  "AQAAABcAAAAAAAAAAAAAAAAAagQAAGFyY2hpdmUvLmZvcm1hdF92ZXJzaW9uUEsBAgAAAAAICAAAAAAAAD93cekCAAAAAgAAABoAAAAAAAAAAAAAAAAA0QQAAGFyY2hpdmUvLnN0b3JhZ2VfYWxpZ2"
  "5tZW50UEsBAgAAAAAICAAAAAAAAIU94xkGAAAABgAAABEAAAAAAAAAAAAAAAAAUgUAAGFyY2hpdmUvYnl0ZW9yZGVyUEsBAgAAAAAICAAAAAAAAC+k00cYAAAAGAAAAA4AAAAAAAAAAAAAAAAA1gUA"
  "AGFyY2hpdmUvZGF0YS8wUEsBAgAAAAAICAAAAAAAAPUdINwIAAAACAAAAA4AAAAAAAAAAAAAAAAAaAYAAGFyY2hpdmUvZGF0YS8xUEsBAgAAAAAICAAAAAAAAJBRBtUIAAAACAAAAA4AAAAAAAAAAA"
  "AAAAAA2AYAAGFyY2hpdmUvZGF0YS8yUEsBAgAAAAAICAAAAAAAAGnfImUIAAAACAAAAA4AAAAAAAAAAAAAAAAAWAcAAGFyY2hpdmUvZGF0YS8zUEsBAgAAAAAICAAAAAAAAMXZ9j8IAAAACAAAAA4A"
  "AAAAAAAAAAAAAAAA2AcAAGFyY2hpdmUvZGF0YS80UEsBAgAAAAAICAAAAAAAAJBRBtUIAAAACAAAAA4AAAAAAAAAAAAAAAAAWAgAAGFyY2hpdmUvZGF0YS81UEsBAgAAAAAICAAAAAAAAHDW528IAA"
  "AACAAAAA4AAAAAAAAAAAAAAAAA2AgAAGFyY2hpdmUvZGF0YS82UEsBAgAAAAAICAAAAAAAAAoSAJEgAAAAIAAAAA4AAAAAAAAAAAAAAAAAWAkAAGFyY2hpdmUvZGF0YS83UEsBAgAAAAAICAAAAAAA"
  "AGyynXsgAAAAIAAAAA4AAAAAAAAAAAAAAAAA8AkAAGFyY2hpdmUvZGF0YS84UEsBAgAAAAAICAAAAAAAAO/UBa8QAAAAEAAAAA4AAAAAAAAAAAAAAAAAcAoAAGFyY2hpdmUvZGF0YS85UEsBAgAAAA"
  "AICAAAAAAAANGeZ1UCAAAAAgAAAA8AAAAAAAAAAAAAAAAA4AoAAGFyY2hpdmUvdmVyc2lvblBLAQIAAAAACAgAAAAAAAAOvyPbKAAAACgAAAAeAAAAAAAAAAAAAAAAAFILAABhcmNoaXZlLy5kYXRh"
  "L3NlcmlhbGl6YXRpb25faWRQSwYGLAAAAAAAAAAeAy0AAAAAAAAAAAAQAAAAAAAAABAAAAAAAAAA6wMAAAAAAAD4CwAAAAAAAFBLBgcAAAAA4w8AAAAAAAABAAAAUEsFBgAAAAAQABAA6wMAAPgLAA"
  "AAAA=="
)
ckpt = torch.load(io.BytesIO(base64.b64decode(REAL_PYTORCH_CHECKPOINT)), map_location="cpu")
m = nn.Sequential(nn.Linear(3, 2), nn.BatchNorm1d(2))
res = m.load_state_dict(ckpt["model"])
m.eval()
out["real_checkpoint"] = [sorted(ckpt.keys()), list(ckpt["model"].keys()), ckpt["epoch"], ckpt["best"], ckpt["tags"], ckpt["ids"], ckpt["half"], ckpt["nested"]["t"],
                          ckpt["nested"]["flag"], ckpt["nested"]["none"] is None, list(ckpt["nested"]["tuple"]), m[0].weight.detach(), m[1].running_mean, int(m[1].num_batches_tracked),
                          m(torch.tensor([[1.0, 2.0, 3.0]])), str(res), ckpt["ids"].dtype.is_floating_point, isinstance(ckpt["model"], OrderedDict)]

class Net(nn.Module):
  def __init__(self):
    super().__init__()
    self.body = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.BatchNorm1d(8))
    self.head = nn.Linear(8, 3)
    self.register_buffer("scale", torch.tensor([2.0]))
    self.register_buffer("scratch", torch.zeros(2), persistent=False)
  def forward(self, x): return self.head(self.body(x)) * self.scale
x = det(5, 4, seed=1)
model = det_init(Net())
opt = torch.optim.Adam(model.parameters(), lr=0.01)
for _ in range(3):
  opt.zero_grad(); F.mse_loss(model(x), torch.zeros(5, 3)).backward(); opt.step()
model.eval()
expected = model(x).detach()
# 1. state_dict to an in-memory buffer
buf = io.BytesIO()
torch.save(model.state_dict(), buf)
buf.seek(0)
restored = Net()
keys = restored.load_state_dict(torch.load(buf))
restored.eval()
# 2. full training checkpoint to a file path
path = os.path.join(tempfile.gettempdir(), "tinygrad_js_ckpt.pt")
torch.save({"epoch": 5, "model_state_dict": model.state_dict(), "optimizer_state_dict": opt.state_dict(), "loss": 0.25, "history": [1.0, 0.5]}, path)
ck = torch.load(path, weights_only=True)
m2 = Net(); m2.load_state_dict(ck["model_state_dict"]); m2.eval()
opt2 = torch.optim.Adam(m2.parameters(), lr=0.5)
opt2.load_state_dict(ck["optimizer_state_dict"])
# continuing training from the checkpoint gives the same result as continuing the original
def cont(mod, o):
  mod.train()
  for _ in range(2):
    o.zero_grad(); F.mse_loss(mod(x), torch.zeros(5, 3)).backward(); o.step()
  mod.eval()
  return mod(x).detach()
after_orig, after_restored = cont(model, opt), cont(m2, opt2)
# 3. plain tensors, strict / non-strict loading, whole-module pickles, deepcopy
tbuf = io.BytesIO(); torch.save([torch.arange(6).view(2, 3), {"w": torch.tensor([1.5, 2.5])}, 3], tbuf); tbuf.seek(0)
tl = torch.load(tbuf)
partial = {k: v for k, v in model.state_dict().items() if k.startswith("body")}
missing = Net().load_state_dict(partial, strict=False)
try: Net().load_state_dict(partial); strict_err = "no error"
except RuntimeError as e: strict_err = "RuntimeError" if "Missing key(s)" in str(e) else str(e)
try: nn.Linear(4, 9).load_state_dict(nn.Linear(4, 8).state_dict()); size_err = "no error"
except RuntimeError as e: size_err = "RuntimeError" if "size mismatch" in str(e) else str(e)
mbuf = io.BytesIO(); torch.save(nn.Linear(2, 2), mbuf); mbuf.seek(0)
whole = torch.load(mbuf, weights_only=False)
clone = copy.deepcopy(model)
with torch.no_grad(): clone.head.weight.zero_()
sd = model.state_dict()
out["roundtrip"] = [restored(x), expected, str(keys), list(sd.keys()), "scratch" in sd, ck["epoch"], ck["loss"], ck["history"], m2(x) if False else after_restored, after_orig,
                    opt2.param_groups[0]["lr"], tl[0], tl[1]["w"], tl[2], sorted(missing.missing_keys), missing.unexpected_keys, strict_err, size_err,
                    type(whole).__name__, list(whole.weight.shape), whole.weight.requires_grad, float(model.head.weight.abs().sum()) > 0, float(clone.head.weight.abs().sum()),
                    [v.requires_grad for v in sd.values()][:2], sd["body.2.num_batches_tracked"].item(), os.path.exists(path)]
emit(**out)
