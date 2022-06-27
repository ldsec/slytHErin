## Results
|   Model   	| Encrypted? 	|   Setting   	| Network Latency 	| Parties    	| Latency(s),Batch                  	 | Thput(im/s),Batch,Parties 	             |
|:---------:	|:----------:	|:-----------:	|:---------------:	|------------	|-------------------------------------|-----------------------------------------|
| SimpleNet 	|     No     	| Centralized 	|        -        	| 1          	| {0.86,36}                        	  | {48.65,163,1}        	                  |
|    NN20   	|     Yes    	|     MPC     	|    Localhost    	| 3 / 5 / 10 	| {292,146} / {328,146} / {392,146} 	 | {0.85,292,3} / {0.8,292,5} / {        	 |
|    NN50   	|     Yes    	|     MPC     	|    Localhost    	| 5 / 10     	| {759,146}, {1007,146}             	 | {0.19,146}        	                     |
|           	|            	|             	|                 	|            	| 	                                   | 	                                       |
### Remarks
- In NN20 we experience a loss of accuracy of ~ 0.6% for approx deg of 63
- In NN50 we experience a loss of accuracy of ~ 4.7% for approx deg of 63