import numpy as np, pandas as pd

T0, T1 = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-09-07")

def kyiv(s):
    return pd.to_datetime(s, utc=True).dt.tz_convert("Europe/Kyiv").dt.tz_localize(None)

def norm(r):
    return "м. Київ" if r in {"Київ", "місто Київ", "м. Київ"} else r

off = pd.read_csv("official_data_uk.csv")
off = pd.DataFrame({"reg": off["oblast"].map(norm), "start": kyiv(off["started_at"]), "end": kyiv(off["finished_at"])})
vol = pd.read_csv("volunteer_data_uk.csv")
naive = vol["naive"].astype(str).str.lower().isin(["true", "1"])
vol = pd.DataFrame({"reg": vol["region"].map(norm), "start": kyiv(vol["started_at"]), "end": kyiv(vol["finished_at"]), "naive": naive})

print("official:", off.start.min(), off.start.max(), "| volunteer:", vol.start.min(), vol.start.max())
for name, d in [("official", off), ("volunteer", vol)]:
    m = d[(d.start >= T0) & (d.start < T1)]
    print(name, "2026 до 07.09 рядків:", len(m))
vm = vol[(vol.start >= T0)]
print("volunteer 2026 naive частка:", round(vm.naive.mean(), 4))
print("volunteer після 07.09:", len(vol[vol.start >= T1]), " naive:", round(vol[vol.start >= T1].naive.mean(), 4))

H = pd.date_range(T0, T1, freq="h", inclusive="left")
def onset_active(d, regs, cap=72):
    d = d[(d.start >= T0 - pd.Timedelta(days=4)) & (d.start < T1)].copy()
    d["end"] = np.minimum(d["end"], d["start"] + pd.Timedelta(hours=cap))
    O = pd.DataFrame(0, index=H, columns=regs, dtype=np.int8)
    A = pd.DataFrame(0, index=H, columns=regs, dtype=np.int8)
    for r, g in d.groupby("reg"):
        if r not in O.columns: continue
        h = g.start.dt.floor("h")
        h = h[(h >= H[0]) & (h <= H[-1])]
        O.loc[h.unique(), r] = 1
        for s, e in zip(g.start, g.end):
            lo, hi = s.floor("h"), (e - pd.Timedelta(seconds=1)).floor("h")
            lo, hi = max(lo, H[0]), min(hi, H[-1])
            if lo <= hi: A.loc[lo:hi, r] = 1
    return O, A

regs = sorted(set(off.reg) & set(vol.reg) - {"Луганська область"})
Oo, Ao = onset_active(off, regs); Ov, Av = onset_active(vol, regs)
rows = []
for r in regs:
    rows.append({"reg": r, "onset_h_off": int(Oo[r].sum()), "onset_h_vol": int(Ov[r].sum()),
                 "daily_corr": np.corrcoef(Oo[r].resample("D").sum(), Ov[r].resample("D").sum())[0,1],
                 "active_h_off": int(Ao[r].sum()), "active_h_vol": int(Av[r].sum()),
                 "active_agree": float((Ao[r] == Av[r]).mean())})
res = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(res.round(3).to_string(index=False))
print("\nСумарно онсет-годин: official", int(Oo.values.sum()), "volunteer", int(Ov.values.sum()), "ratio vol/off", round(Ov.values.sum()/Oo.values.sum(), 3))
print("Сумарно активних годин: official", int(Ao.values.sum()), "volunteer", int(Av.values.sum()))
# по місяцях
mo = pd.DataFrame({"off": Oo.sum(axis=1).resample("MS").sum(), "vol": Ov.sum(axis=1).resample("MS").sum()})
mo["ratio"] = mo["vol"] / mo["off"]
print("\nОнсет-години по місяцях:\n", mo.round(3).to_string())
