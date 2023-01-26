"""
Module from the assignments for UC Berkeley's Deep RL course.
Modify frozen_lake.py for our purpose(2022.11.08)
"""

import numpy as np
import discrete_env

class ParentalLeave(discrete_env.DiscreteEnv):

    def __init__(self, ):
        self.l_for_promo = 5  # {대리 승진까지, 과장 승진까지, 차장 승진까지} 소요되는 기간. 0이면 승진 누락 상태
        self.n_position = 4  # {사원, 대리, 과장, 차장}, 직급
        self.n_options = 2  # {0번, 1번}, 육아휴직 남은 횟수
        self.max_age = 10  # {1,2,3,4,5,6,7,8,9,10살 이상}, 아이의 나이
        self.max_years = 15
        # 상태 = (승진 남은 기간, 직급, 육아휴직 남은 횟수, 아이 나이, 총 근속 연수)

        #아이의 수 = 1이라 할 떄
        #if opt == 1 and age == 10:
            #opt = opt-1

        add_years = True  # 휴직기간 가산

        nS = self.l_for_promo * self.n_position * self.n_options * self.max_age * self.max_years  # 상태의 가능한 경우의 수
        nA = 2
        action_name = {0: "근무", 1: "휴직 사용"}

        P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        def to_s(left, pos, opt, age, time):
            s_idx = 0
            s_idx += time
            s_idx += age * self.max_years
            s_idx += opt * (self.max_years * self.max_age)
            s_idx += pos * (self.max_years * self.max_age * self.n_options) #state(0,3,0,0,0)에서 걸린다.
            s_idx += left * (self.max_years * self.max_age * self.n_options * self.n_position)
            return s_idx


        # 첫 상태 결정 확률
        isd = np.zeros(nS)
        # 예를 들어, 첫 상태가 (4년 남음, 사원, 2번 남음, 1살, 0 시점)
        isd[to_s(4, 0, 2, 0, 0)] = 1.0

        # 휴직기간 가산, 확률 100퍼센트로 승진으로 생각
        for left in range(self.l_for_promo):
            for pos in range(self.n_position):
                for opt in range(self.n_options):
                    for age in range(self.max_age):
                        # n_age 정의하기
                        n_age = min(age + 1, self.max_age-1)
                        for time in range(self.max_years):
                            c_s = to_s(left, pos, opt, age, time)
                            done = False
                            if time + 1 == self.max_years:
                                done = True #next value를 받지 않는다.
                                time -= 1
                            for a in range(nA):
                                rew = self.get_reward(pos, age, action_name[a])  # r(s,a)
                                li = P[c_s][a]  # 빈 list []

                                if action_name[a] == '근무':
                                    if left == 1: # 승진 대상, pos == 3 일 수 없음
                                    #if pos in [0,1,2,3]:
                                        promo_prob = self.get_promo_prob(pos, opt)
                                        # 승진 시
                                        n_s = to_s(self.l_for_promo-1, min(pos + 1, self.n_position-1), opt, n_age, time + 1)
                                        li.append((promo_prob, n_s, rew, done))  # n_s는 다음 상태
                                        # 미승진 시
                                        n_s = to_s(left-1, pos, opt, n_age, time + 1)
                                        li.append((1.0-promo_prob, n_s, rew, done))  # n_s는 다음 상태
                                    #elif pos == 3:
                                       # n_s = to_s(left-1,pos,opt,n_age,time+1) ,이러면 pos = 3일때도 4년 근무한다고 해야함.
                                       #li.append(1.0,n_s,rew,done))


                                    elif left == 0: # 승진 누락자는 무조건 승진
                                        n_s = to_s(self.l_for_promo-1, min(pos + 1, self.n_position-1), opt, n_age, time + 1) #pos가 3일때 4로 가버린다.
                                        li.append((1.0, n_s, rew, done))  # n_s는 다음 상태
                                    elif left > 1: # 승진 대상 아님
                                        n_s = to_s(left-1, pos, opt, n_age, time + 1)
                                        li.append((1.0, n_s, rew, done))  # n_s는 다음 상태
                                elif action_name[a] == '휴직 사용':
                                    n_opt = opt - 1
                                    if left == 1: # 승진 대상,
                                        if add_years == True:  # 승진기간 가산 시
                                            promo_prob = self.get_promo_prob(pos, n_opt)
                                            # 승진 시
                                            n_s = to_s(self.l_for_promo-1,min(pos + 1, self.n_position-1), n_opt, n_age, time + 1)
                                            li.append((promo_prob, n_s, rew, done))  # n_s는 다음 상태
                                            # 미승진 시
                                            n_s = to_s(left-1, pos, n_opt, n_age, time + 1)
                                            li.append((1.0-promo_prob, n_s, rew, done))  # n_s는 다음 상태
                                        else:  # 승진기간 미가산 시
                                            n_s = to_s(left, pos, n_opt, n_age, time + 1)
                                            li.append((1.0, n_s, rew, done))  # n_s는 다음 상태
                                    elif left == 0: # 승진 누락자는 무조건 승진
                                        n_s = to_s(self.l_for_promo - 1, min(pos + 1, self.n_position-1), n_opt, n_age, time + 1)
                                        li.append((1.0, n_s, rew, done))  # n_s는 다음 상태
                                    elif left > 0:
                                        if add_years == True:  # 승진기간 가산 시
                                            n_s = to_s(left - 1, pos, n_opt, n_age, time + 1)
                                            li.append((1.0, n_s, rew, done))  # n_s는 다음 상태
                                        else:  # 승진기간 미가산 시
                                            n_s = to_s(left, pos, n_opt, n_age, time + 1)
                                            li.append((1.0, n_s, rew, done))  # n_s는 다음 상태

        super(ParentalLeave, self).__init__(nS, nA, P, isd)

    def get_reward(self, pos, age, action):
        if action == '근무':
            return self.salary(pos)
        elif action == '휴직 사용':# age에 따라서 매우 큰 penalty 줘야할 수도
            #if age == 10:
                #return -10000
            return max(840, min(self.salary(pos) * 0.8, 2100))

    # 직급별 salary 함수
    def salary(self, pos):
        if pos == 0:
            return 4000
        elif pos == 1:
            return 4000 * 1.5
        elif pos == 2:
            return 4000 * 1.5 * 1.5
        elif pos == 3:
            return 4000 * 1.5 * 1.5 * 1.5
        # action 기준 r(s,a)

    # 승진 확률 함수
    def get_promo_prob(self, pos, opt):
        if pos == 0 or pos == 1:  # 낮은직급일 떄
            if opt == 0:
                return 0.5  # 직급이 낮고 육아휴직 횟수 0회 남으면 0.5 승진 가능성
            elif opt == 1:
                return 0.7
            else:
                return 1.0
        elif pos == 2:
            if opt == 0:
                return 0.3
            elif opt == 1:
                return 0.6
            else:
                return 1.0
        elif pos == 3:
            return 1.0
        #pos = 0,1,2,3,4로 둔 후 if pos ==3: return 1.0
        #

#max_age로 age증가 제한, 아이 나이 초기화 시점:
#아이 나이는

# if __name__ == '__main__':
#     env = ParentalLeave()
#     # VI
#     vi_model = Value_iteration(env, discount_rate = 1.0)
#     policy, value = vi_model.solve(max_iter=10000)
#     print("Policy: ", policy)
#     print("Value: ", value)
#
#     # PI
#     pi_model = Policy_iteration(env, discount_rate = 1.0)
#     policy, value = pi_model.solve(max_iter=10000)
#     print("Policy: ", policy)
#     print("Value: ", value)

#QSA index
#to_s값 잘못 만들었을 가능성
#s_index가 넘어가도록 만들었을 간으성