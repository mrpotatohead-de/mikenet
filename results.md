Mikenet Speedup Test (with/without BLIS)
===

[TOC]

## BLIS-optimized-Mikenet Installation Guide
https://aionchip.computing.ncku.edu.tw:3001/NVZEZzksSfOqkeThjnnmYw?view

## Tests
### Test on Workstation
| Optimization Strategy | Oral          | Reading-OP | Reading-OS |
| --------------------- | ------------- | ---------- | ---------- |
| BLIS                  | 42055.57     |  378971.071006          | 379720.477832           |
| raw                   | 162780.217463 | 1338915.415579           | 1318528.694773           |
| Ratio                      |  3.87             |  3.53          |  3.47          |

BLIS Oral
42,055.57 seconds = 11.682102778 hours
| seed\iteration | 400000   | 1200000   | 2000000 | Accumulated |
| -------------- | -------- | --------- | ------- | ----------- |
| 1              | 910.23   | 2687.11   | 4792.59 | 8,389.93    |
| 2              | 890.49   | 2804.11   | 4678.33 | 8,372.93    |
| 3              | 939.17   | 2729.22   | 4692.51 | 8,360.90    |
| 4              | 934.13   | 2770.70   | 4698.18 | 8,403.01    |
| 5              | 975.53   | 2776.98   | 4776.29        |  8,528.80           |
| Accumulated    | 4,649.55 | 13,768.12 | 23,637.90        | 42,055.57            |

Raw Oral
| seed\iteration | 400000       | 1200000      | 2000000      | Accumulated   |
| -------------- | ------------ | ------------ | ------------ | ------------- |
| 1              | 3564.045376  | 10732.414477 | 18058.261702 | 32354.721555              |
| 2              | 3579.412549  | 10834.960454 | 18155.780626 | 32570.153629              |
| 3              | 3650.555744  | 10856.221637 | 17920.183307 | 32426.960688              |
| 4              | 3613.284542  | 10686.609808 | 18407.695542 | 32707.589892              |
| 5              | 3600.394984  | 10891.771084 | 18228.625631 | 32720.791699              |
| Accumulated    | 18007.693195 | 54001.977460 | 90770.546808 | 162780.217463 |

BLIS Reading-OP
| seed\iteration | LP  | MP  | HP  | Accumulated |
| -------------- | --- | --- | --- | ----------- |
| 1              | 25075.267899    | 25392.693597    | 24709.865539    |  75177.827035           |
| 2              | 25307.428780    | 25253.831675    | 25375.968262    | 75937.228717            |
| 3              | 25311.029444    | 25322.508045    | 25435.688793    | 76069.226282            |
| 4              | 25308.562473    | 25322.253172    | 25216.035759    | 75846.851404            |
| 5              | 25298.935996    | 25251.231432    |  25389.770140   | 75939.937568            |
| Accumulated    |  126301.224592   | 126542.517921    | 126127.328493    | 378971.071006            |

BLIS Reading-OS
| seed\iteration | LP  | MP  | HP  | Accumulated |
| -------------- | --- | --- | --- | ----------- |
| 1              | 25383.705081    | 25234.153131    | 25317.553773    |             |
| 2              | 25455.950220    | 25228.713017    | 25251.591948    |             |
| 3              | 25281.227463    | 25213.934599    | 25473.899298    |             |
| 4              | 25126.783035    | 25479.144798    | 25325.965646    |             |
| 5              | 25365.778878    | 25443.116138    | 25138.960807    |             |
| Accumulated    |  126613.444677   | 126599.061683   |     | 379720.477832            |


Raw Reading-OS
| seed\iteration | LP  | MP  | HP  | Accumulated |
| -------------- | --- | --- | --- | ----------- |
| 1              | 86639.303407    | 86384.330970    | 100409.908063    |             |
| 2              | 86053.053443    | 86437.458491    | 88629.676428    |             |
| 3              | 86435.559008    | 86338.132185    | 88628.307332    |             |
| 4              | 86812.510176    | 86264.968588    | 88549.733323    |             |
| 5              | 86372.828703    | 86118.256672    | 88454.667984    |             |
| Accumulated    |     |     |     |  1318528.694773           |



Raw Reading-OP
| seed\iteration | LP  | MP  | HP  | Accumulated |
| -------------- | --- | --- | --- | ----------- |
| 1              | 86332.070560    | 86262.262996    | 117705.552281    |             |
| 2              | 86262.245189    | 87017.551861    | 88551.085056    |             |
| 3              | 86558.171438    | 86181.768421    | 88797.188311    |             |
| 4              | 86425.982281    | 86171.590839    | 88769.809689    |             |
| 5              | 86692.972748    | 88616.977727    | 88570.186182    |             |
| Accumulated    |     |     |     | 1338915.415579            |

```
sh ./train.sh > log.202210192326.txt


910.23user 0.19system 15:10.59elapsed 99%CPU (0avgtext+0avgdata 72196maxresident)k
0inputs+144088outputs (0major+18190minor)pagefaults 0swaps
890.49user 0.23system 14:50.90elapsed 99%CPU (0avgtext+0avgdata 72048maxresident)k
0inputs+143784outputs (0major+18190minor)pagefaults 0swaps
939.17user 0.17system 15:39.52elapsed 99%CPU (0avgtext+0avgdata 72144maxresident)k
0inputs+143808outputs (0major+18191minor)pagefaults 0swaps
934.13user 0.14system 15:34.43elapsed 99%CPU (0avgtext+0avgdata 72052maxresident)k
0inputs+143856outputs (0major+18186minor)pagefaults 0swaps
975.53user 0.16system 16:15.88elapsed 99%CPU (0avgtext+0avgdata 72036maxresident)k
0inputs+143816outputs (0major+18196minor)pagefaults 0swaps

2687.11user 0.39system 44:47.99elapsed 99%CPU (0avgtext+0avgdata 72084maxresident)k
0inputs+148776outputs (0major+18194minor)pagefaults 0swaps
2804.11user 0.39system 46:45.00elapsed 99%CPU (0avgtext+0avgdata 72060maxresident)k
0inputs+149024outputs (0major+18186minor)pagefaults 0swaps
2729.22user 0.33system 45:30.15elapsed 99%CPU (0avgtext+0avgdata 72052maxresident)k
0inputs+148808outputs (0major+18188minor)pagefaults 0swaps
2770.70user 0.39system 46:11.57elapsed 99%CPU (0avgtext+0avgdata 72152maxresident)k
0inputs+145520outputs (0major+18192minor)pagefaults 0swaps
2776.98user 0.39system 46:17.98elapsed 99%CPU (0avgtext+0avgdata 72144maxresident)k
0inputs+145536outputs (0major+18189minor)pagefaults 0swaps

4792.59user 0.53system 1:19:54elapsed 99%CPU (0avgtext+0avgdata 72084maxresident)k
0inputs+147064outputs (0major+18186minor)pagefaults 0swaps
4678.33user 0.55system 1:17:59elapsed 99%CPU (0avgtext+0avgdata 72080maxresident)k
328inputs+147064outputs (0major+18190minor)pagefaults 0swaps
4692.51user 0.52system 1:18:13elapsed 99%CPU (0avgtext+0avgdata 72088maxresident)k
0inputs+147056outputs (0major+18199minor)pagefaults 0swaps
4698.18user 0.53system 1:18:19elapsed 99%CPU (0avgtext+0avgdata 72152maxresident)k
0inputs+147056outputs (0major+18191minor)pagefaults 0swaps
4776.29user 0.67system 1:19:37elapsed 99%CPU (0avgtext+0avgdata 72060maxresident)k
0inputs+147096outputs (0major+18185minor)pagefaults 0swaps
```
### Test on Qoca
| Optimization Strategy | Oral          | Reading-OP | Reading-OS |
| --------------------- | ------------- | ---------- | ---------- |
| BLIS                  | 42122.157952  | 195557.052972           | 196722.925318           |
| raw                   | 179144.641663 |            |            |
| ratio                    |  4.25             |            |            |

BLIS Oral
| seed\iteration | 400000 | 1200000 | 2000000 | Accumulated |
| -------------- | ------ | ------- | ------- | ----------- |
| 1              | 955.822650       |  2780.123664       | 4716.894699        |  8452.841013           |
| 2              |  931.037247      |  2821.468407       | 4727.452567        |  8479.958221           |
| 3              | 918.349664       |  2796.522664       | 4714.018742        |  8428.891070           |
| 4              | 928.809177       |  2808.230041       | 4620.505168        |  8357.544386           |
| 5              | 928.357675       |  2814.052521       | 4660.513066        |  8402.923262           |
| Accumulated    |  4662.376413      |   14020.397297      |    23439.384242     |  42122.157952           |

Raw Oral
| seed\iteration | 400000      | 1200000      | 2000000      | Accumulated |
| -------------- | ----------- | ------------ | ------------ | ----------- |
| 1              | 3994.276251 | 11931.675605 | 19900.877176 |  35826.829032           |
| 2              | 3992.377815 | 11938.022376 | 19899.121007 |  35829.521198           |
| 3              | 3989.899878 | 11940.442845 | 19929.828118 |  35860.170841           |
| 4              | 3980.636368 | 11937.670517 | 19931.758421 |  35850.065306           |
| 5              | 3977.289723 | 11974.860714 | 19825.904849 |  35778.055286           |
| Accumulated    | 19934.480035            |   59722.672057           |   99487.489571           |  179144.641663           |

BLIS Reading-OP
| seed\iteration | LP  | MP  | HP  | Accumulated |
| -------------- | --- | --- | --- | ----------- |
| 1              |  13140.211777   | 13197.749736    | 12915.128858    |             |
| 2              |  12984.043227   | 13144.534552    | 13165.608205    |             |
| 3              |  12774.025285   | 13027.900843    | 12880.492294    |             |
| 4              |  13095.416780   | 12900.853105    | 12891.978735    |             |
| 5              |  13122.307327   | 12897.196599    | 13419.605649    |             |
| Accumulated    |     |     |     | 195557.052972            |

BLIS Reading-OS
| seed\iteration | LP  | MP  | HP  | Accumulated |
| -------------- | --- | --- | --- | ----------- |
| 1              | 13348.406155    | 13060.682593    | 12912.398872    |             |
| 2              | 12939.911780    | 12993.244633    | 12983.886952    |             |
| 3              | 13203.160901    | 12987.894204    | 13060.964494    |             |
| 4              | 13318.253586    | 13258.916636    | 13470.785251    |             |
| 5              | 13101.851912    | 12867.252466    | 13215.314883    |             |
| Accumulated    |     |     |     |  196722.925318           |

Raw Reading-OP
| seed\iteration | LP  | MP  | HP  | Accumulated |
| -------------- | --- | --- | --- | ----------- |
| 1              | 66484.164340    | 66487.489849    |     |             |
| 2              | 66809.354662    | 66464.888438    |     |             |
| 3              | 66466.002342    |     |     |             |
| 4              | 66191.153189    |     |     |             |
| 5              | 66447.587867    |     |     |             |
| Accumulated    |     |     |     |             |

Raw Reading-OS
| seed\iteration | LP  | MP  | HP  | Accumulated |
| -------------- | --- | --- | --- | ----------- |
| 1              | 66145.051251    | 66728.502430    |     |             |
| 2              | 66305.953729    | 66298.378843    |     |             |
| 3              | 66283.905088    |     |     |             |
| 4              | 66690.470409    |     |     |             |
| 5              | 66661.129664    |     |     |             |
| Accumulated    |     |     |     |             |





