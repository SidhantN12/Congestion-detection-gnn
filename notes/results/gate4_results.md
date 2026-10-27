# Gate 4: every held-out sample and channel

All regressions fit training designs only. Kendall is exact here. Hotspots use ratio ≥ 0.9, top-|S| predictions and fractional cutoff-tie credit; precision equals recall by construction.

| sample | method | channel | spearman | kendall | kendall_sampled | precision | recall | roc_auc | mae | rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| aes__u35_s2 | Capacity-only | H | 0.0804 | 0.0611 | False | 0.0433 | 0.0433 | 0.5316 | 0.1457 | 0.1735 |
| aes__u35_s2 | Capacity-only | V | 0.0706 | 0.0546 | False | 0.2635 | 0.2635 | 0.5321 | 0.1764 | 0.2046 |
| aes__u35_s2 | Density-only | H | 0.8011 | 0.6065 | False | 0.1702 | 0.1702 | 0.8341 | 0.0946 | 0.1263 |
| aes__u35_s2 | Density-only | V | 0.7798 | 0.5940 | False | 0.6109 | 0.6109 | 0.8469 | 0.1072 | 0.1423 |
| aes__u35_s2 | RUDY | H | 0.8731 | 0.6948 | False | 0.2723 | 0.2723 | 0.8641 | 0.4846 | 0.4948 |
| aes__u35_s2 | RUDY | V | 0.8641 | 0.6930 | False | 0.6772 | 0.6772 | 0.8927 | 0.5005 | 0.5154 |
| aes__u35_s2 | CongestionGNN | H | 0.7925 | 0.5991 | False | 0.1191 | 0.1191 | 0.8237 | 0.1339 | 0.1577 |
| aes__u35_s2 | CongestionGNN | V | 0.7805 | 0.5937 | False | 0.6006 | 0.6006 | 0.8485 | 0.1630 | 0.1862 |
| aes__u40_s3 | Capacity-only | H | 0.0862 | 0.0649 | False | 0.0580 | 0.0580 | 0.5259 | 0.1342 | 0.1595 |
| aes__u40_s3 | Capacity-only | V | 0.0902 | 0.0703 | False | 0.3247 | 0.3247 | 0.5518 | 0.1634 | 0.1870 |
| aes__u40_s3 | Density-only | H | 0.7911 | 0.5985 | False | 0.2457 | 0.2457 | 0.8429 | 0.0974 | 0.1313 |
| aes__u40_s3 | Density-only | V | 0.7209 | 0.5388 | False | 0.6380 | 0.6380 | 0.8337 | 0.1107 | 0.1503 |
| aes__u40_s3 | RUDY | H | 0.8500 | 0.6709 | False | 0.3391 | 0.3391 | 0.8770 | 0.4922 | 0.5033 |
| aes__u40_s3 | RUDY | V | 0.8360 | 0.6642 | False | 0.7149 | 0.7149 | 0.8875 | 0.5016 | 0.5181 |
| aes__u40_s3 | CongestionGNN | H | 0.7513 | 0.5630 | False | 0.2042 | 0.2042 | 0.8249 | 0.1254 | 0.1493 |
| aes__u40_s3 | CongestionGNN | V | 0.7280 | 0.5484 | False | 0.6310 | 0.6310 | 0.8327 | 0.1553 | 0.1771 |
| aes__u45_s1 | Capacity-only | H | 0.1621 | 0.1233 | False | 0.1088 | 0.1088 | 0.5435 | 0.1307 | 0.1547 |
| aes__u45_s1 | Capacity-only | V | 0.1258 | 0.0978 | False | 0.3902 | 0.3902 | 0.5554 | 0.1590 | 0.1817 |
| aes__u45_s1 | Density-only | H | 0.7492 | 0.5534 | False | 0.2453 | 0.2453 | 0.8136 | 0.1008 | 0.1398 |
| aes__u45_s1 | Density-only | V | 0.6803 | 0.5015 | False | 0.6536 | 0.6536 | 0.8126 | 0.1100 | 0.1509 |
| aes__u45_s1 | RUDY | H | 0.8548 | 0.6751 | False | 0.4533 | 0.4533 | 0.8862 | 0.4974 | 0.5082 |
| aes__u45_s1 | RUDY | V | 0.8204 | 0.6497 | False | 0.7511 | 0.7511 | 0.8858 | 0.5097 | 0.5257 |
| aes__u45_s1 | CongestionGNN | H | 0.7270 | 0.5415 | False | 0.2547 | 0.2547 | 0.7906 | 0.1251 | 0.1475 |
| aes__u45_s1 | CongestionGNN | V | 0.6840 | 0.5081 | False | 0.6555 | 0.6555 | 0.8144 | 0.1532 | 0.1741 |

RUDY leads the GNN in aggregate ranking and hotspot metrics in both channels. Density-only beats it horizontally and is close vertically. Capacity-only is a weak floor. These results support learning beyond capacity but do not establish a GNN advantage over analytical routing demand. RUDY predicts an uncalibrated demand proxy, so its MAE/RMSE assess absolute calibration as well as spatial prediction; its ranking remains useful despite scale error.
