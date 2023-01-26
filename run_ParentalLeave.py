from VIandPI_v2 import Value_iteration, Policy_iteration
from ParentalLeave_v2 import ParentalLeave

import openpyxl
import numpy as np
import pandas as pd

ls = []
for a in range(5):#left for promo
    for b in range(4):#pos
        for c in range(2):#opt
            for d in range(10):#age
                for e in range(15):#time
                    ls.append([e,d,c,b,a])
df = pd.DataFrame(ls)
df.columns = ['Work_time','Child_age','Option','Position','Left_for_promo']



wb = openpyxl.Workbook()
sheet = wb.active
wb.save('test.xlsx')


if __name__ == '__main__':
    env = ParentalLeave()
    # VI
    vi_model = Value_iteration(env, discount_rate = 1.0)
    policy, value = vi_model.solve(max_iter=10000)
    print("Policy: ", policy)
    print("Value: ", value)
    #state 정의할 때 나온 index를 별 해당되도록 엑셀파일이던 해서(0~5999까지 해당되는 state 만들기)
    #policy: action >> state별로 해서 나온다. (0,0,0,0,1), (0,0,0,1,1) 이런식으로
    #무조건 +의 사람인데(휴직중이라도 이득을 보긴 하는데) 그럼 여기서 육아휴직 쓸 수 없을 때 done을 때려버린다.


    # PI
    pi_model = Policy_iteration(env, discount_rate = 1.0)
    policy, value = pi_model.solve(max_iter=10000)
    print("Policy: ", policy)
    print("Value: ", value)

    vl = value
    pol = policy
    df_vl = pd.DataFrame(vl)
    df_pol = pd.DataFrame(pol)

df_1 = pd.concat([df,df_vl],axis=1)
df_2 = pd.concat([df_1,df_pol],axis=1)


df_2.columns = ['Work_time','Child_age','Option','Position','Left_for_promo','Value','Policy']
print(df_2)

df_2.to_excel('Parental_Leave.xlsx',index=False)

