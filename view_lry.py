from lry import LRYRecord


rec = LRYRecord(
    path_lry="/Users/steven/Project/CardioDelinear/data/lry/小鼠1-16173016-converted-2000HZ.LRY",
    path_r_on="/Users/steven/Project/CardioDelinear/data/lry/小鼠1-16173016-R_on.txt",
    path_r_off="/Users/steven/Project/CardioDelinear/data/lry/小鼠1-16173016-R_off.txt",
)
rec.read()
ann = rec.read_annotations()
rec.plot(lead=0, start=rec.start_sec, duration=10.0)  # 画前10秒