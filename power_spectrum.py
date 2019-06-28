import numpy as np
import matplotlib.pyplot as plt
import yt
import os

def sample(k,p):
    master_sum=0.0
    cnt = 0.0
    for x in range(11):
        for y in range(11):
            for z in range(11):
                master_sum+=(cal(k,x,y,z,p)/10**6)
    return master_sum

def cal(k,x,y,z,p):
    sum = 0.0
    cnt = 0
    kx_l=np.arange(max(0,x-k),min((x+k+1),11))
    ky_l=np.arange(max(0,y-k),min((y+k+1),11))
    kz_l=np.arange(max(0,z-k),min((z+k+1),11))
    if (k==0):
        return p[x,y,z]
    for kx in kx_l:
        for ky in ky_l:
            for  kz in kz_l:
                if ((kx-x)*(kx-x) + (ky-y)*(ky-y) + (kz-z)*(kz-z))<=k*k and ((kx-x)*(kx-x)+(ky-y)*(ky-y)+(kz-z)*(kz-z))>(k-1)*(k-1):
                    sum = sum + p[kx,ky,kz]
                    cnt +=1
    return sum/cnt


unit_base = {'UnitLength_in_cm'         : 3.08568e+21,
             'UnitMass_in_g'            :   1.989e+43,
             'UnitVelocity_in_cm_per_s' :      100000}
bbox_lim = 100 #mpc

bbox = [[0,bbox_lim],
        [0,bbox_lim],
        [0,bbox_lim]]

path = 'Test/snapshot_020'
#path = 'Outputs2/snapshots/snapshot_000'
ds = yt.load(path,unit_base=unit_base,bounding_box=bbox)

ad = ds.all_data()

particle_x = ad['all','particle_position_x']

particle_y = ad['all','particle_position_y']

particle_z =  ad['all','particle_position_z']

particle_position = ad['all', 'particle_position']

cube_side = 10

rho = np.zeros(shape= [11,11,11])
i = 0


while(i<len(particle_position)):
    pos_x = float(particle_x[i])
    pos_y = float(particle_y[i])
    pos_z = float(particle_z[i])
    loc_x = pos_x/cube_side
    loc_y = pos_y/cube_side
    loc_z = pos_z/cube_side
    x = int(pos_x)
    y=int(pos_y)
    z=int(pos_z)
    rho[int(loc_x),int(loc_y),int(loc_z)] += (x+1-pos_x)*(y+1-pos_y)*(z+1-pos_z)
    rho[int(loc_x)+1,int(loc_y),int(loc_z)] += (pos_x-x)*(y+1-pos_y)*(z+1-pos_z)
    rho[int(loc_x),int(loc_y)+1,int(loc_z)] += (x+1-pos_x)*(pos_y-y)*(z+1-pos_z)
    rho[int(loc_x),int(loc_y),int(loc_z)+1] += (x+1-pos_x)*(y+1-pos_y)*(pos_z-z)

    rho[int(loc_x)+1,int(loc_y)+1,int(loc_z)] += (pos_x-x)*(pos_y-y)*(z+1-pos_z)
    rho[int(loc_x),int(loc_y)+1,int(loc_z)+1]+= (x+1-pos_x)*(pos_y-y)*(pos_z-z)
    rho[int(loc_x)+1,int(loc_y),int(loc_z)+1] += (pos_x-x)*(y+1-pos_y)*(pos_z-z)
    rho[int(loc_x)+1,int(loc_y)+1,int(loc_z)+1] += (pos_x-x)*(pos_y-y)*(pos_z-z)
    i = i+1

temp=0
cnt=0
for a in rho:
    for b in a:
        for c in b:
            temp+=c
            cnt+=1

rho = (rho/temp)*cnt -1
p = abs(np.fft.fftn(rho))
p *=p
amplitude = []
power = []
wave = []

for k in range(6):
    wave.append(k)
    power.append(sample(k,p))
    print("**")
    print(k)

#power = ((power/total)*n -1)

plt.plot(wave,power)

plt.show()
