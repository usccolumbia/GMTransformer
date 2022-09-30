import pandas as pd
import moses
import argparse
import selfies as sf

parser = argparse.ArgumentParser()

parser.add_argument('--input', default='sample.txt',
	                    help='path to selfie token file')											      
parser.add_argument('--valid',
	                    help='path to validation file')
parser.add_argument('--output', default='output_smiles.txt',
	                    help='smiles file')

args = parser.parse_args()
name1 = args.input.split('.')[0]
df=pd.read_csv(args.input,header=None)


d2 = []
for s in df.iloc[:,0]:
    if '<' in s:
        continue
    s = ''.join(s.split(' '))
    
    s = sf.decoder(s)
    d2.append(s)

print(d2)


df1 = pd.DataFrame(d2,columns=['formula'])
df1.to_csv(f'{args.output}',header=None,index=None)


# metrics = moses.get_all_metrics(d2)
# df2 = pd.DataFrame.from_dict(metrics,orient="index")
# df2.to_csv(f'{name1}_metrics.csv',header=None)

