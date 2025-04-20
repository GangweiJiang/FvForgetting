import json
import numpy as np

# expname='fexp/ni_t11_e10_42'
expnames=[
"ni_seq2_h20_42",
"ni_seq2_h21_42",
"ni_seq2_h22_42",
"ni_seq2_h23_42",
"ni_seq2_h24_42",
"ni_seq2_h25_42",
"ni_seq2_h26_42",
"ni_seq2_h27_42",
]

for expname in expnames:
    print(expname)
    # expname="fexp/ni_t31_iv_0_002_h0_h002_l3_fv_42"

    # types = ['t','g','i']
    types = ['general','icl','train', ]
    # types = ['general' ]
    # types = ['general','icl']
    # types = ['train']
    # types = ['corr']
    stages = ['6'] if 'trace' in expname else ['5']
    # stages = ['1','2','3','4']
    # stages = ['2']
    # stages = ['1','2','3','4']
    # 22.34 27.22 89.0 100.0 60.0 61.33 
    alls = []
    for t in types:
        # print(t,t,t,t,t,t,t,t,t,t,t)
        results = {}
        # results = {
        #     'icl_coqa': [],
        #     'icl_hellaswag': [],
        #     'coqa': [],
        #     'hellaswag': []
        # }
        for s in stages:
            try:
                filepath = f'/disk/jianggangwei/fv_guided_traning/results/{expname}/results_{t}_f{s}.json'
                with open(filepath, 'r') as f:
                    data = json.load(f)    
                    for k, v in data.items():
                        try:
                            if k in results:
                                results[k].append( np.mean(np.array(v['score'])))
                            else:
                                results[k] = [ np.mean(np.array(v['score']))]
                            # results[]
                            # print(k, np.mean(np.array(v['score'])))
                        except Exception as e:
                            tt=1
            except:
                continue
        # for keys in results.keys():
        #     print(keys, end=" ")
        # print('')
        for i in range(len(stages)):
            ds = []
            for key in results.keys():
                # print(round(results[key][i]*100, 2), end=" ")
                ds.append(round(results[key][i]*100, 2))
            # print('')
            alls.append(round(np.mean(ds),2))
    print(alls)