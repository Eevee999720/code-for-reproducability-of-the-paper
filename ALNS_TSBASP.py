import numpy as np
import time as t

import random
import math




import copy


theta1 = 1.5
theta2 = 1.2
theta3 = 0.8
initial_temperature = 500
min_temperature = 1
cooling_rate = 0.99
b = 0.5
init_weight=1


outer_iterations = 10
lb=[0.5,0.5,0.5]
ub=[10,10,10]






n_dim = 40
qu = 12
at = [1, 1, 2, 2, 18, 22, 35, 38, 50, 54, 78, 78, 86, 90, 94, 102, 110, 122, 122, 152, 131, 131, 132, 132, 148, 152, 165, 168, 180, 184, 208, 208, 216, 220, 224, 232, 240, 252, 252, 282]
lt = [21, 26, 16, 25, 51, 41, 46, 54, 79, 88, 100, 104, 112, 112, 105, 122, 129, 149, 136, 183, 151, 156, 146, 155, 181, 171, 176, 184, 209, 218, 230, 234, 242, 242, 235, 252, 259, 279, 266, 313]
sl = [3,7,  3, 8, 6, 4, 7, 8, 6, 6, 4, 5, 7, 8, 5, 4, 4, 7, 3, 4, 3,7,  3, 8, 6, 4, 7, 8, 6, 6, 4, 5, 7, 8, 5, 4, 4, 7, 3, 4]

hv =[18, 96, 11, 113, 113, 25, 18, 27, 114, 105, 30, 66, 109, 84, 22, 31, 36, 108, 13, 49, 18, 96, 11, 113, 113, 25, 18, 27, 114, 105, 30, 66, 109, 84, 22, 31, 36, 108, 13, 49]
min_equipment = [ 2, 3, 2, 3, 3, 2, 3, 3, 3, 3, 2, 2, 3, 3, 2, 2, 2, 3, 2, 2, 2, 3, 2, 3, 3, 2, 3, 3, 3, 3, 2, 2, 3, 3, 2, 2, 2, 3, 2, 2]
max_equipment = [2, 6, 2, 7, 5, 3, 6, 7, 5, 5, 3, 4, 6, 7, 4, 3, 3, 6, 2, 3, 2, 6, 2, 7, 5, 3, 6, 7, 5, 5, 3, 4, 6, 7, 4, 3, 3, 6, 2, 3]
pb = [13, 2, 14, 11, 12, 13, 9, 5, 6, 16, 4, 17, 15, 10, 19, 16, 12, 6, 18, 1, 13, 2, 14, 11, 12, 13, 9, 5, 6, 16, 4, 17, 15, 10, 19, 16, 12, 6, 18, 1]
c1 = 2
c2 = 10
c3 =1
c4=5
c5=5
alph=1
lb1=[0 for i in range (n_dim)]+min_equipment
ub1=[24-sl[i] for i in range (n_dim)]+max_equipment
bn = 24
tn=5
tw = 600
RN=  [1,3, 1, 3, 2,    1, 3, 3, 2, 2,    1, 2 ,3, 3, 2,    1 , 1,3, 1 ,1, 1,3, 1, 3, 2,    1, 3, 3, 2, 2,    1, 2 ,3, 3, 2,    1 , 1,3, 1 ,1 ]
RS=[[3,2,2,1,1],[100,100,3,2,1],[3,2,2,1,1],[100,100,3,2,1],[100,3,2,2,1],
    [3,2,2,1,1],[100,100,3,2,1],[100,100,3,2,1],[100,3,2,2,1],[100,3,2,2,1],
    [3,2,2,1,1],[100,3,2,2,1],[100,100,3,2,1],[100,100,3,2,1],[100,3,2,2,1],
[3,2,2,1,1],[3,2,2,1,1],[100,100,3,2,1],[3,2,2,1,1],[3,2,2,1,1],
[3,2,2,1,1],[100,100,3,2,1],[3,2,2,1,1],[100,100,3,2,1],[100,3,2,2,1],
    [3,2,2,1,1],[100,100,3,2,1],[100,100,3,2,1],[100,3,2,2,1],[100,3,2,2,1],
    [3,2,2,1,1],[100,3,2,2,1],[100,100,3,2,1],[100,100,3,2,1],[100,3,2,2,1],
    [3,2,2,1,1],[3,2,2,1,1],[100,100,3,2,1],[3,2,2,1,1],[3,2,2,1,1]]
CN=[1,3,  1, 3, 2, 1, 3, 3, 2, 2, 1, 2 ,3, 3, 2, 1 , 1,2, 1 ,1, 1,3,  1, 3, 2, 1, 3, 3, 2, 2, 1, 2 ,3, 3, 2, 1 , 1,2, 1 ,1]
CS=[[3,2,2,1,1],[100,100,3,2,1],[3,2,2,1,1],[100,100,3,2,1],[100,3,2,2,1],
    [3,2,2,1,1],[100,100,3,2,1],[100,100,3,2,1],[100,3,2,2,1],[100,3,2,2,1],
    [3,2,2,1,1],[100,3,2,2,1],[100,100,3,2,1],[100,100,3,2,1],[100,3,2,2,1],
[3,2,2,1,1],[3,2,2,1,1],[100,100,3,2,1],[3,2,2,1,1],[3,2,2,1,1],
[3,2,2,1,1],[100,100,3,2,1],[3,2,2,1,1],[100,100,3,2,1],[100,3,2,2,1],
    [3,2,2,1,1],[100,100,3,2,1],[100,100,3,2,1],[100,3,2,2,1],[100,3,2,2,1],
    [3,2,2,1,1],[100,3,2,2,1],[100,100,3,2,1],[100,100,3,2,1],[100,3,2,2,1],
    [3,2,2,1,1],[3,2,2,1,1],[100,100,3,2,1],[3,2,2,1,1],[3,2,2,1,1]]
safe=1







class Sou():
    def __init__(self):
        self.rt=[]
        self.ct=[]
        self.p = []
        self.q=[]
        self.sr=[]
        self.s=[]
        self.h = []
        self.e = []
        self.ce=[]



def func(sou):

    t_berth=[ce_i - a_i for ce_i, a_i in zip(sou.ce, at)]
    f1=c1*np.sum(t_berth)

    t_late = [0 if td_i <= leave_time_i else abs(td_i - leave_time_i) for td_i, leave_time_i in zip(sou.ce, lt)]

    t_wait_r = [sr_i - arrival_i for sr_i, arrival_i in zip(sou.sr, at)]
    eh=[s_i + h_i for s_i, h_i in zip(sou.s, sou.h)]
    t_wait_c = [e_i - eh_i for e_i, eh_i in zip(sou.e, eh)]
    f2=c2*(np.sum(t_late)+np.sum(t_wait_r)+np.sum(t_wait_c))

    l_p=[abs(p_i-pb_i) for p_i, pb_i in zip(sou.p, pb)]
    f3=c3*np.sum(l_p)
    #
    t_rtug=[i*RS[index][i-1] for index, i in enumerate(sou.rt)]
    t_ctug = [i*CS[index][i - 1] for index, i in enumerate(sou.ct)]
    f4=c4*(np.sum(t_rtug)+np.sum(t_ctug))


    f5=c5*np.sum(np.multiply(sou.h, sou.q))





    return f1+f2+f3+f4+f5


def func2(sou):

    t_berth=[ce_i - a_i for ce_i, a_i in zip(sou.ce, at)]
    f1=c1*np.sum(t_berth)

    t_late = [0 if td_i <= leave_time_i else abs(td_i - leave_time_i) for td_i, leave_time_i in zip(sou.ce, lt)]

    t_wait_r = [sr_i - arrival_i for sr_i, arrival_i in zip(sou.sr, at)]
    eh=[s_i + h_i for s_i, h_i in zip(sou.s, sou.h)]
    t_wait_c = [e_i - eh_i for e_i, eh_i in zip(sou.e, eh)]
    f2=c2*(np.sum(t_late)+np.sum(t_wait_r)+np.sum(t_wait_c))

    l_p=[abs(p_i-pb_i) for p_i, pb_i in zip(sou.p, pb)]
    f3=c3*np.sum(l_p)

    t_rtug=[i*RS[index][i-1] for index, i in enumerate(sou.rt)]
    t_ctug = [i*CS[index][i - 1] for index, i in enumerate(sou.ct)]
    f4=c4*(np.sum(t_rtug)+np.sum(t_ctug))


    f5=c5*np.sum(np.multiply(sou.h, sou.q))
    print("f1",f1)
    print("f2", f2)
    print("f3", f3)
    print("f4", f4)
    print("f5", f5)



    return f1+f2+f3+f4+f5


def reverse_tug_monitor(k,s,list):
    for x in range(tn,RN[k]-1,-1):
        temp_x_h=RS[k][x-1]
        if np.all(5-list[s-temp_x_h:s]>=x):
            return s-temp_x_h
    return 0

def dl_reverse_tug_monitor(k,s,list,rt):
    temp_x_h = RS[k][rt - 1]
    if np.all(5 - list[s - temp_x_h:s] >= rt):
        return s - temp_x_h
    return 0

def cal_r_tug(k,s,list):
    for x in range(tn,RN[k]-1,-1):
        temp_x_h=RS[k][x-1]
        if np.all(5-list[s-temp_x_h:s]>=x):
            return s-temp_x_h,x

def is_leave_t(k,s,list):
    s=int(s)
    if tn-list[s]<CN[k]:
        return 0
    else:
        temp = np.array(range(int(CN[k]), int(tn - list[s] + 1)))


        for i in temp:
            temph = RS[k][i-1]
            if np.all(5 - list[s:s + temph] >= CN[k]):

                return 1

        return 0

def monitor_leave_tug(k,s,list):

    temp = np.array(range(CN[k], tn - list[s] + 1))
    atug = []
    for i in temp:
        temph = RS[k][i - 1]
        if np.all(5 - list[s:s + temph] >= CN[k]):

            atug.append(i)
    return atug

def solt_fill(des,p,s,e):
    slot=np.zeros((bn, tw))
    os=[]
    for i in range(n_dim):
        if i not in des:
            os.append(i)
            slot[int(p[i]):int(p[i]) + int(sl[i]),
            int(s[i]):int(e[i])+safe] = 1
    return os,slot

def tuglist_fill(sou,des):
    list_t = np.zeros(tw, dtype=int)

    for i in range(n_dim):
        if i not in des:
            list_t[sou.sr[i]:sou.s[i]] = [element + sou.rt[i] for element in
                                          list_t[sou.sr[i]:sou.s[i]]]
            list_t[sou.e[i]:sou.ce[i]] = [element + sou.rt[i] for element in
                                          list_t[sou.e[i]:sou.ce[i]]]


    return list_t



def cal_seq_t(sou,des):#时间优先


    service_order = sorted(des, key=lambda i: sou.s[i] - sou.sr[i] + sou.e[i] - sou.s[i] - sou.h[i] + lt[i] - sou.e[i], reverse=True)

    return service_order


def cal_seq_b(des,p):


    indexed_arrival_times = [(index, arrival_time) for index, arrival_time in enumerate(des)]


    indexed_arrival_times.sort(key=lambda x: (-abs(p[x[1]] - pb[x[1]]), random.random()))


    service_order = []


    for index, _ in indexed_arrival_times:
        service_order.append(_)


    return service_order


def cal_seq_q(des,q):


    indexed_arrival_times = [(index, arrival_time) for index, arrival_time in enumerate(des)]


    indexed_arrival_times.sort(key=lambda x: (-(max_equipment[x[1]] - q[x[1]]), random.random()))


    service_order = []


    for index, _ in indexed_arrival_times:
        service_order.append(_)


    return service_order

def cal_seq_tug(des,rt,ct):


    indexed_arrival_times = [(index, arrival_time) for index, arrival_time in enumerate(des)]


    indexed_arrival_times.sort(key=lambda x: (-abs(ct[x[1]]+rt[x[1]]-RN[x[1]]-CN[x[1]]), random.random()))


    service_order = []


    for index, _ in indexed_arrival_times:
        service_order.append(_)

    # 返回船舶服务的顺序
    return service_order


def find_closest_ships(x_coords, y_coords):
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)

    num_ships = len(x_coords)
    distances = np.zeros((num_ships, num_ships))
    temp_chu=5


    for i in range(num_ships):
        for j in range(num_ships):
            distances[i, j] = np.sqrt((x_coords[i] - x_coords[j]) ** 2 + ((y_coords[i] - y_coords[j]) ** 2)/temp_chu)

    min_distance = float('inf')
    closest_pair = (None, None)

    for i in range(num_ships):
        for j in range(i + 1, num_ships):
            if distances[i][j] < min_distance:
                min_distance = distances[i][j]
                closest_pair = [i, j]

    return closest_pair

def find_quayp_ships(sou, n=2):

    ship_diffs = [(i, max_quay - actual_quay) for i, (max_quay, actual_quay) in
                  enumerate(zip(max_equipment, sou.q))]

    # 根据差距从大到小排序并取前n个
    largest_diff_ship_ids = [ship_id for ship_id, _ in sorted(ship_diffs, key=lambda x:( x[1], random.random()), reverse=True)[:n]]

    return largest_diff_ship_ids

def find_berthp_ships(sou, n=2):

    ship_diffs = [(i, abs(ber-pbe)) for i, (ber, pbe) in
                  enumerate(zip(sou.p, pb))]


    largest_diff_ship_ids = [ship_id for ship_id, _ in sorted(ship_diffs, key=lambda x:( x[1], random.random()), reverse=True)[:n]]

    return largest_diff_ship_ids


def have_berth(x,y,oq):
    ab=[]

    for i in range(bn-sl[x]+1):
        if np.all(oq[i:i + sl[x], y] == 0):
            ab.append(i)
    return ab

def pef_have_berth(x,y,oq):
    ab=[]
    start = math.ceil(pb[x] - sl[x] / 2)
    end = math.floor(pb[x] + sl[x] / 2)
    temp = list(range(start, end + 1))

    for i in temp:
        if np.all(oq[i:i + sl[x], y] == 0):  # 切片操作是左开右闭的
            ab.append(i)
    return ab


def find_nearest_point(array, x):
    distances = np.abs(array - x)
    min_distance = np.min(distances)
    nearest_points = array[distances == min_distance]
    return np.random.choice(nearest_points)


def cal_bst():
    pass

def cal_st(k,oo,s):
    s[k]=at[k]
    matrix=oo.copy()
    while len(have_berth(k, s[k], matrix)) == 0:
        s[k]+=1

    return s[k]

def cal_st1(k,oo):
    s=at[k]
    matrix=oo.copy()
    while len(have_berth(k, s, matrix)) == 0:
        s+=1

    return s

def cal_ts(k,x,y,matrix):
    l=1
    while np.all(matrix[x:x + sl[k], y+l+1] == 0) and l<100:
        l+=1
    return l,(hv[k])/l


def span_monitor(sv, x,y,l,p,q,s,h):
    aqn = qu
    idx = []
    idx1 = []

    for k in sv:
        if abs(y - s[k] + 0.5 * (l - h[k])) < 0.5 * (l + h[k]):
            aqn -= q[k]
            idx.append(k)
        else:
            idx1.append(k)

    if len(idx) > 0:
        for z in idx:
            if p[z] > x:
                for m in idx1:#
                    if p[m] > p[z] and s[m]+h[m]<=y and s[m]+h[m]>s[z]:
                        aqn -= q[m]
            else:
                for m in idx1:  #
                    if p[m] < p[z] and s[m] + h[m] <= y and s[m] + h[m] > s[z]:
                        aqn -= q[m]

    return aqn

def point_monitor(sv, x,y,p,q,s,h):
    aqn = qu
    idx = []
    idx1 = []

    for k in sv:
        if abs(y-(s[k]+0.5*h[k]))< 0.5 * h[k] or y==s[k]:
            aqn -= q[k]
            idx.append(k)
        else:
            idx1.append(k)

    if len(idx) > 0:
        for z in idx:
            if p[z] > x:
                for m in idx1:#
                    if p[m] > p[z] and s[m]+h[m]<=y and s[m]+h[m]>s[z]:
                        aqn -= q[m]
            else:
                for m in idx1:  #
                    if p[m] < p[z] and s[m] + h[m] <= y and s[m] + h[m] > s[z]:
                        aqn -= q[m]

    return aqn


def npoint_monitor(sv, x,y,p,q,s,h,l,qm):
    l=int(l)
    for i in range(y+1,y+l):
        if point_monitor(sv,x,i,p,q,s,h)<qm:
            return 0
    return 1




def tp_insert_ship1(k, sv, matrix, p, q, s, h,list_t):
    temp_s = cal_st1(k, matrix)

    while True:
        while reverse_tug_monitor(k,temp_s,list_t)<at[k]:
            temp_s += 1


        temp_sr,temp_rtug=cal_r_tug(k,temp_s,list_t)
        if np.array(have_berth(k, temp_s, matrix)).size > 0:
            os = temp_s
            temp_B = np.array(have_berth(k, temp_s, matrix))
            hand_seq_B = sorted(temp_B, key=lambda i: abs(i - pb[k]))
            for x in hand_seq_B:

                temp_l, temp_qm = cal_ts(k, x, temp_s, matrix)
                temp_qa = point_monitor(sv, x, temp_s, p, q, s, h)
                if temp_qa >= temp_qm and temp_qa >= min_equipment[k]:


                    start = max(min_equipment[k], int(np.ceil(temp_qm)))

                    end = min(temp_qa, max_equipment[k])

                    AQ = list(range(start, end + 1))

                    AQ = sorted(AQ, reverse=True)

                    for z in AQ:
                        temp_h = int(np.ceil((hv[k]) / z))
                        if npoint_monitor(sv,x,temp_s, p, q, s, h,temp_h,z)==1:
                            temp_e = temp_s + temp_h
                            while temp_e <= os + temp_l:
                                if is_leave_t(k,temp_e,list_t)==1:
                                    atug=monitor_leave_tug(k,temp_e,list_t)
                                    temp_ctug=max(atug)
                                    temp_ct=CS[k][temp_ctug-1]
                                    temp_ce=temp_e+temp_ct
                                    return temp_rtug,temp_ctug,x, z,temp_sr, temp_s, temp_h, temp_e,temp_ce
                                else:
                                    temp_e += 1

            temp_s += 1

        else:
            temp_s += 1


def bp_insert_ship1(k, sv, matrix, p, q, s, h,list_t):

    rt=random.randint(RN[k],5)
    temp_s = cal_st1(k, matrix)
    temp_rh=RS[k][rt - 1]


    while True:
        while dl_reverse_tug_monitor(k,temp_s,list_t,rt)<at[k]:
            temp_s += 1


        temp_sr,temp_rtug=temp_s-temp_rh,rt
        if np.array(have_berth(k, temp_s, matrix)).size > 0:
            os = temp_s
            temp_B = np.array(have_berth(k, temp_s, matrix))
            hand_seq_B = sorted(temp_B, key=lambda i: abs(i - pb[k]))
            for x in hand_seq_B:

                temp_l, temp_qm = cal_ts(k, x, temp_s, matrix)
                temp_qa = point_monitor(sv, x, temp_s, p, q, s, h)
                if temp_qa >= temp_qm and temp_qa >= min_equipment[k]:


                    start = max(min_equipment[k], int(np.ceil(temp_qm)))

                    end = min(temp_qa, max_equipment[k])

                    AQ = list(range(start, end + 1))

                    random.shuffle(AQ)

                    for z in AQ:
                        temp_h = int(np.ceil((hv[k]) / z))
                        if npoint_monitor(sv,x,temp_s, p, q, s, h,temp_h,z)==1:
                            temp_e = temp_s + temp_h
                            while temp_e <= os + temp_l:
                                if is_leave_t(k,temp_e,list_t)==1:
                                    atug=monitor_leave_tug(k,temp_e,list_t)
                                    temp_ctug=random.choice(atug)
                                    temp_ct=CS[k][temp_ctug-1]
                                    temp_ce=temp_e+temp_ct
                                    return temp_rtug,temp_ctug,x, z,temp_sr, temp_s, temp_h, temp_e,temp_ce
                                else:
                                    temp_e += 1

            temp_s += 1

        else:
            temp_s += 1


def tugp_insert_ship1(k, sv, matrix, p, q, s, h, list_t):
    temp_s = cal_st1(k, matrix)

    while True:
        while reverse_tug_monitor(k, temp_s, list_t) < at[k]:
            temp_s += 1


        temp_sr, temp_rtug = cal_r_tug(k, temp_s, list_t)
        if np.array(have_berth(k, temp_s, matrix)).size > 0:
            os = temp_s
            temp_B = np.array(have_berth(k, temp_s, matrix))
            temp_B = temp_B[temp_B > 0]
            hand_seq_B = temp_B
            np.random.shuffle(hand_seq_B)
            for x in hand_seq_B:
                temp_l, temp_qm = cal_ts(k, x, temp_s, matrix)
                temp_qa = point_monitor(sv, x, temp_s, p, q, s, h)
                if temp_qa >= temp_qm and temp_qa >= min_equipment[k]:


                    start = max(min_equipment[k], int(np.ceil(temp_qm)))

                    end = min(temp_qa, max_equipment[k])

                    AQ = list(range(start, end + 1))

                    random.shuffle(AQ)

                    for z in AQ:
                        temp_h = int(np.ceil((hv[k]) / z))
                        if npoint_monitor(sv,x,temp_s, p, q, s, h,temp_h,z)==1 :
                            temp_e = temp_s + temp_h
                            while temp_e <= os + temp_l:
                                if is_leave_t(k, temp_e, list_t) == 1:
                                    atug = monitor_leave_tug(k, temp_e, list_t)
                                    temp_ctug = max(atug)
                                    temp_ct = CS[k][temp_ctug - 1]
                                    temp_ce = temp_e + temp_ct
                                    return temp_rtug, temp_ctug, x, z, temp_sr, temp_s, temp_h, temp_e, temp_ce
                                else:
                                    temp_e += 1

            temp_s += 1

        else:
            temp_s += 1


def qp_insert_ship1(k, sv, matrix, p, q, s, h, list_t):

    rt = random.randint(RN[k], 5)
    temp_s = cal_st1(k, matrix)
    temp_rh = RS[k][rt - 1]

    while True:
        while dl_reverse_tug_monitor(k, temp_s, list_t, rt) < at[k]:
            temp_s += 1


        temp_sr, temp_rtug = temp_s - temp_rh, rt
        if np.array(have_berth(k, temp_s, matrix)).size > 0:
            os = temp_s
            temp_B = np.array(have_berth(k, temp_s, matrix))
            temp_B = temp_B[temp_B > 0]
            hand_seq_B = temp_B
            np.random.shuffle(hand_seq_B)
            for x in hand_seq_B:

                temp_l, temp_qm = cal_ts(k, x, temp_s, matrix)
                temp_qa = point_monitor(sv, x, temp_s, p, q, s, h)
                if temp_qa >= temp_qm and temp_qa >= min_equipment[k]:


                    start = max(min_equipment[k], int(np.ceil(temp_qm)))

                    end = min(temp_qa, max_equipment[k])

                    AQ = list(range(start, end + 1))

                    AQ = sorted(AQ, reverse=True)

                    for z in AQ:
                        temp_h = int(np.ceil((hv[k]) / z))
                        if npoint_monitor(sv,x,temp_s, p, q, s, h,temp_h,z)==1:
                            temp_e = temp_s + temp_h
                            while temp_e <= os + temp_l:
                                if is_leave_t(k, temp_e, list_t) == 1:
                                    atug = monitor_leave_tug(k, temp_e, list_t)
                                    temp_ctug = random.choice(atug)
                                    temp_ct = CS[k][temp_ctug - 1]
                                    temp_ce = temp_e + temp_ct
                                    return temp_rtug, temp_ctug, x, z, temp_sr, temp_s, temp_h, temp_e, temp_ce
                                else:
                                    temp_e += 1

            temp_s += 1

        else:
            temp_s += 1


def rand_insert_ship1(k, sv, matrix, p, q, s, h, list_t):

    rt = random.randint(RN[k], 5)
    temp_s = cal_st1(k, matrix)
    temp_rh = RS[k][rt - 1]

    while True:
        while dl_reverse_tug_monitor(k, temp_s, list_t, rt) < at[k]:
            temp_s += 1


        temp_sr, temp_rtug = temp_s - temp_rh, rt
        if np.array(have_berth(k, temp_s, matrix)).size > 0:
            os = temp_s
            temp_B = np.array(have_berth(k, temp_s, matrix))
            temp_B = temp_B[temp_B > 0]
            hand_seq_B = temp_B
            np.random.shuffle(hand_seq_B)
            for x in hand_seq_B:

                temp_l, temp_qm = cal_ts(k, x, temp_s, matrix)
                temp_qa = point_monitor(sv, x, temp_s, p, q, s, h)
                if temp_qa >= temp_qm and temp_qa >= min_equipment[k]:

                    start = max(min_equipment[k], int(np.ceil(temp_qm)))

                    end = min(temp_qa, max_equipment[k])

                    AQ = list(range(start, end + 1))

                    random.shuffle(AQ)

                    for z in AQ:
                        temp_h = int(np.ceil((hv[k]) / z))
                        if npoint_monitor(sv,x,temp_s, p, q, s, h,temp_h,z)==1:
                            temp_e = temp_s + temp_h
                            while temp_e <= os + temp_l:
                                if is_leave_t(k, temp_e, list_t) == 1:
                                    atug = monitor_leave_tug(k, temp_e, list_t)
                                    temp_ctug = random.choice(atug)
                                    temp_ct = CS[k][temp_ctug - 1]
                                    temp_ce = temp_e + temp_ct
                                    return temp_rtug, temp_ctug, x, z, temp_sr, temp_s, temp_h, temp_e, temp_ce
                                else:
                                    temp_e += 1

            temp_s += 1

        else:
            temp_s += 1


def random_destory(sou):
    return random.sample(range(n_dim), 2)

def berth_destory(sou):
    l_p=[abs(p_i-pb_i) for p_i, pb_i in zip(sou.p, pb)]
    indices = sorted(range(len(l_p)), key=lambda x: (l_p[x],random.random()), reverse=True)
    return indices[:2]

def time_destory(sou):
    t_late = [0 if td_i <= leave_time_i else abs(td_i - leave_time_i) for td_i, leave_time_i in zip(sou.e, lt)]

    t_wait_r = [sr_i - arrival_i for sr_i, arrival_i in zip(sou.sr, at)]
    eh = [s_i + h_i for s_i, h_i in zip(sou.s, sou.h)]
    t_wait_c = [e_i - eh_i for e_i, eh_i in zip(sou.e, eh)]
    t_all=[t_late[i]+t_wait_r[i]+t_wait_c[i] for i in range(n_dim)]

    indices = sorted(range(len(t_all)), key=lambda x: (t_all[x],random.random()), reverse=True)
    return indices[:2]

def similar_destory(sou):
    x,y=sou.p,sou.h
    return find_closest_ships(x,y)

def quay_destory(sou):
    return find_quayp_ships(sou)

def tug_destory(sou):
    l_t1 = [abs(p_i - pb_i) for p_i, pb_i in zip(sou.rt, RN)]
    l_t2 = [abs(p_i - pb_i) for p_i, pb_i in zip(sou.ct, CN)]
    indices = sorted(range(len(l_t1)), key=lambda x:(l_t1+l_t2,random.random()), reverse=True)
    return indices[:2]



def random_repair(sou,des):
    sv, matrix = solt_fill(des, sou.p, sou.s, sou.e)
    list = tuglist_fill(sou, des)

    random.shuffle(des)


    for k in des:
        sou.rt[k], sou.ct[k], sou.p[k], sou.q[k], sou.sr[k], sou.s[k], sou.h[k], sou.e[k], sou.ce[k] = rand_insert_ship1(
            k, sv, matrix, sou.p, sou.q, sou.s, sou.h, list)
        matrix[int(sou.p[k]):int(sou.p[k]) + int(sl[k]), int(sou.s[k]):int(sou.e[k]) + safe] = 1
        list[sou.sr[k]:sou.s[k]] = [element + sou.rt[k] for element in list[sou.sr[k]:sou.s[k]]]
        list[sou.e[k]:sou.ce[k]] = [element + sou.rt[k] for element in list[sou.e[k]:sou.ce[k]]]

        sv.append(k)

    return sou


def timep_repair(sou,des):

    sv, matrix = solt_fill(des, sou.p, sou.s, sou.e)
    list=tuglist_fill(sou,des)
    seq = cal_seq_t(sou,des)

    for k in seq:
        sou.rt[k],sou.ct[k],sou.p[k],sou.q[k],sou.sr[k],sou.s[k],sou.h[k],sou.e[k],sou.ce[k] = tp_insert_ship1(k,sv,matrix,sou.p,sou.q,sou.s,sou.h,list)  # 求得k的分配信息并更新解
        matrix[int(sou.p[k]):int(sou.p[k]) + int(sl[k]), int(sou.s[k]):int(sou.e[k])+safe] = 1
        list[sou.sr[k]:sou.s[k]] = [element + sou.rt[k] for element in list[sou.sr[k]:sou.s[k]]]
        list[sou.e[k]:sou.ce[k]] = [element + sou.rt[k] for element in list[sou.e[k]:sou.ce[k]]]

        sv.append(k)

    return sou

def quayp_repair(sou,des):
    sv, matrix = solt_fill(des, sou.p, sou.s, sou.e)
    list = tuglist_fill(sou, des)
    seq = cal_seq_q(des, sou.p)

    for k in seq:
        sou.rt[k], sou.ct[k], sou.p[k], sou.q[k], sou.sr[k], sou.s[k], sou.h[k], sou.e[k], sou.ce[k] = qp_insert_ship1(
            k, sv, matrix, sou.p, sou.q, sou.s, sou.h, list)
        matrix[int(sou.p[k]):int(sou.p[k]) + int(sl[k]), int(sou.s[k]):int(sou.e[k]) + safe] = 1
        list[sou.sr[k]:sou.s[k]] = [element + sou.rt[k] for element in list[sou.sr[k]:sou.s[k]]]
        list[sou.e[k]:sou.ce[k]] = [element + sou.rt[k] for element in list[sou.e[k]:sou.ce[k]]]

        sv.append(k)

    return sou



def berthp_repair(sou,des):
    sv, matrix = solt_fill(des, sou.p, sou.s, sou.e)
    list = tuglist_fill(sou, des)
    seq = cal_seq_b(des,sou.p)

    for k in seq:
        sou.rt[k], sou.ct[k], sou.p[k], sou.q[k], sou.sr[k], sou.s[k], sou.h[k], sou.e[k], sou.ce[k] = bp_insert_ship1(
            k, sv, matrix, sou.p, sou.q, sou.s, sou.h, list)
        matrix[int(sou.p[k]):int(sou.p[k]) + int(sl[k]), int(sou.s[k]):int(sou.e[k]) + safe] = 1
        list[sou.sr[k]:sou.s[k]] = [element + sou.rt[k] for element in list[sou.sr[k]:sou.s[k]]]
        list[sou.e[k]:sou.ce[k]] = [element + sou.rt[k] for element in list[sou.e[k]:sou.ce[k]]]

        sv.append(k)

    return sou

def tugp_repair(sou,des):
    sv, matrix = solt_fill(des, sou.p, sou.s, sou.e)
    list = tuglist_fill(sou, des)
    seq = cal_seq_tug(des, sou.rt,sou.ct)

    for k in seq:
        sou.rt[k], sou.ct[k], sou.p[k], sou.q[k], sou.sr[k], sou.s[k], sou.h[k], sou.e[k], sou.ce[k] = tugp_insert_ship1(
            k, sv, matrix, sou.p, sou.q, sou.s, sou.h, list)
        matrix[int(sou.p[k]):int(sou.p[k]) + int(sl[k]), int(sou.s[k]):int(sou.e[k]) + safe] = 1
        list[sou.sr[k]:sou.s[k]] = [element + sou.rt[k] for element in list[sou.sr[k]:sou.s[k]]]
        list[sou.e[k]:sou.ce[k]] = [element + sou.rt[k] for element in list[sou.e[k]:sou.ce[k]]]

        sv.append(k)

    return sou





def gen_ini_solution(sou):

    matrix = np.zeros((bn, tw))
    list_t = np.zeros(tw, dtype=int)

    sv=[]#表示
    for i in range(n_dim):



        temp_rtug, temp_ctug, x, z, temp_sr, temp_s, temp_h, temp_e, temp_ce\
            =rand_insert_ship1(i,sv,matrix,sou.p,sou.q,sou.s,sou.h,list_t)
        sou.rt.append(temp_rtug),sou.ct.append(temp_ctug),sou.p.append(x),sou.q.append(z),
        sou.sr.append(temp_sr),sou.s.append(temp_s),sou.h.append(temp_h),sou.e.append(temp_e),sou.ce.append(temp_ce)
        sv.append(i)
        matrix[int(sou.p[i]):int(sou.p[i]) + int(sl[i]), int(sou.s[i]):int(sou.e[i])+safe] = 1
        list_t[sou.sr[i]:sou.s[i]] = [element + sou.rt[i] for element in list_t[sou.sr[i]:sou.s[i]]]
        list_t[sou.e[i]:sou.ce[i]] = [element + sou.rt[i] for element in list_t[sou.e[i]:sou.ce[i]]]


    return sou

def adaptive_select(weights):
    total = sum(weights)
    probabilities = [w / total for w in weights]
    return random.choices(range(len(weights)), probabilities)[0]


print("star running")
t1 = t.time()

sou=Sou()
current_solution = gen_ini_solution(sou)
best_solution = current_solution
current_cost=func(best_solution)
print("cost of ini_solution:",current_cost)
best_cost = current_cost
history_best_obj = []
history_best_obj.append(best_cost)

# 初始化操作集合和评分
destroy_operators = [random_destory, berth_destory, time_destory,quay_destory,tug_destory]#, berth_destory, time_destory,quay_destory,tug_destory
repair_operators = [random_repair,berthp_repair,quayp_repair,tugp_repair,random_repair]
destroy_weights = [init_weight for _ in destroy_operators]
repair_weights = [init_weight for _ in repair_operators]
destroy_scores = [0 for _ in destroy_operators]
repair_scores = [0 for _ in repair_operators]
destroy_selection_count = [1 for _ in destroy_operators]
repair_selection_count = [1 for _ in repair_operators]



for outer_iter in range(outer_iterations):
    print("iteration:",outer_iter)
    temperature = initial_temperature

    while temperature > min_temperature:

        destroy_index = adaptive_select(destroy_weights)
        repair_index = adaptive_select(repair_weights)

        destroy_operator = destroy_operators[destroy_index]
        repair_operator = repair_operators[repair_index]



        destroyed_solution = destroy_operator(copy.deepcopy(current_solution))
        new_solution = repair_operator(copy.deepcopy(current_solution),destroyed_solution)



        new_cost = func(new_solution)


        accept = False
        if new_cost < current_cost:
            accept = True
            if new_cost < best_cost:
                best_solution = new_solution
                best_cost = new_cost
                destroy_scores[destroy_index] += theta1
                repair_scores[repair_index] += theta1
            else:
                destroy_scores[destroy_index] += theta2
                repair_scores[repair_index] += theta2
        else:
            delta = new_cost - current_cost
            acceptance_probability = math.exp(-delta / temperature)
            if random.random() < acceptance_probability:

                accept = True
                destroy_scores[destroy_index] += theta3
                repair_scores[repair_index] += theta3


        if accept:
            current_solution = new_solution
            current_cost = new_cost


        destroy_selection_count[destroy_index] += 1
        repair_selection_count[repair_index] += 1


        temperature *= cooling_rate



    print("destroy_scores:",destroy_scores)
    print("repair_scores:",repair_scores)
    print("destroy_weights:",destroy_weights)
    print("repair_weights:",repair_weights)
    print("Best solution found has cost:", best_cost)
    for i in range(len(destroy_weights)):
        destroy_weights[i] = b * destroy_weights[i] + (1 - b) * (destroy_scores[i] / destroy_selection_count[i])
    for i in range(len(repair_weights)):
        repair_weights[i] = b * repair_weights[i] + (1 - b) * (repair_scores[i] / repair_selection_count[i])

    # 重置评分和选择次数
    destroy_scores = [0 for _ in destroy_operators]
    repair_scores = [0 for _ in repair_operators]
    destroy_selection_count = [1 for _ in destroy_operators]
    repair_selection_count = [1 for _ in repair_operators]

# 输出最优解
print("Best solution found has cost:", best_cost)

t2 = t.time()
print("running time:",t2-t1)
print(best_solution.rt)
print(best_solution.ct)
print(best_solution.p)
print(best_solution.q)
print(best_solution.sr)
print(best_solution.s)
print(best_solution.h)
print(best_solution.e)
print(best_solution.ce)
print(func2(best_solution))







