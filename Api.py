import kaggle
import pandas as pd
import os


def kagglesLoad(ar,file,msg):
    # pd.DataFrame({'Path': fitness(ar.values).ravel()},dtype=int).to_csv(file,index=False) # needs edits with new fitness

    command = f'kaggle competitions submit -c traveling-santa-2018-prime-paths -f {file} -m "{msg}"'
    # print(command)
    os.system(command)
