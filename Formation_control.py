import numpy as np
import matplotlib.pyplot as plt
import time
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
from numpy.linalg import inv
import math

class Robot:
    def __init__(self):
        self.plot_flag=1
        self.vtext_line_counter=0
        self.wtext_line_counter=0        
        self.demonstrated_traj=1        
        self.N=3  
        self.R=15        
        self.r_in = .4
        self.r_out = .65
        self.T = 0.1    
        self.dim =2        
        self.p_star=1*np.array([[0.0, 0.5, 0.0],
                                [-0.5, 0.0, 0.5 ]])   
        
        self.u_t=np.zeros((self.dim,1))
        self.v_t = np.zeros((self.dim,1))
        self.v_t_prev = np.zeros((self.dim,1))
        self.p_t = np.zeros((self.dim,1))

        
        self.p = np.zeros((self.dim,self.N))    
        self.v = np.zeros((self.dim,self.N))
        self.u = np.zeros((self.dim,self.N))
        self.e = np.zeros((self.dim,self.N))
        self.u1 = np.zeros((self.dim,self.N))
        self.u2 = np.zeros((self.dim,self.N))
        self.u3 = np.zeros((self.dim,self.N))
        self.u4 = np.zeros((self.dim,self.N))
        self.u5 = np.zeros((self.dim,self.N))
        self.u6 = np.zeros((self.dim,self.N))
        self.u7 = np.zeros((self.dim,self.N))       
        self.v_err=np.zeros((self.dim,1))
        self.e_err=np.zeros((self.dim,1))
        self.Q_coll=.08
        self.Q_conn=.08        
        self.p[0,0]=0
        self.p[1,0]=-0.5
        self.p[0,1]=0
        self.p[1,1]=0
        self.p[0,2]=0
        self.p[1,2]=0.5        
        for j in np.arange(0, self.N,1):            
            self.v[0,j]=0.0
            self.v[1,j]=0.0
       
        self.a=np.array([[0.0, 1.0, 1.0],
                         [1.0, 0.0, 1.0],
                         [1.0, 1.0, 0.0]])

        self.d_jk=np.array([[0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0]])
        self.d_12_hist=np.array([])
        self.d_13_hist=np.array([])
        self.d_23_hist=np.array([])
        self.t1=0;
        self.d_min=100;
        self.d_max=0;

        self.d_min_hist=np.array([])
        self.d_max_hist=np.array([])
        self.pt_x_hist=np.array([])
        self.pt_y_hist=np.array([])
        self.pa_x_hist=np.array([])
        self.pa_y_hist=np.array([])
        self.pb_x_hist=np.array([])
        self.pb_y_hist=np.array([])
        self.pc_x_hist=np.array([])
        self.pc_y_hist=np.array([])
        self.e_err_x_hist=np.array([])
        self.e_err_y_hist=np.array([])
        self.v_err_x_hist=np.array([])
        self.v_err_y_hist=np.array([])
        self.t_hist=np.array([])
        self.e1_x_hist=np.array([])
        self.e1_y_hist=np.array([])
        self.e2_x_hist=np.array([])
        self.e2_y_hist=np.array([])
        self.e3_x_hist=np.array([])
        self.e3_y_hist=np.array([])
        self.t1=0        
        self.p_nh = np.zeros((self.dim,self.N))   
        self.p_nh[0,0]=0
        self.p_nh[1,0]=-.6
        self.p_nh[0,1]=0
        self.p_nh[1,1]=0
        self.p_nh[0,2]=0
        self.p_nh[1,2]=.6
        self.v_nh = np.zeros((self.dim,self.N))    
        self.theta_nh = np.zeros((self.N,1))
        self.e_nh = np.zeros((self.dim,self.N))
        self.l=.2   
        self.Q_nh_1=np.zeros((2,2))
        self.Q_nh_2=np.zeros((2,2))
        self.Q_nh_3=np.zeros((2,2))

        self.pa_x_nh_hist=np.array([])
        self.pa_y_nh_hist=np.array([])
        self.pb_x_nh_hist=np.array([])
        self.pb_y_nh_hist=np.array([])
        self.pc_x_nh_hist=np.array([])
        self.pc_y_nh_hist=np.array([])

        self.e_nh1_x_hist=np.array([])
        self.e_nh2_x_hist=np.array([])
        self.e_nh3_x_hist=np.array([])

        self.e_nh1_y_hist=np.array([])
        self.e_nh2_y_hist=np.array([])
        self.e_nh3_y_hist=np.array([])

        self.va_nh_hist=np.array([])
        self.vb_nh_hist=np.array([])
        self.vc_nh_hist=np.array([])

        self.wa_nh_hist=np.array([])
        self.wb_nh_hist=np.array([])
        self.wc_nh_hist=np.array([])       
        self.vtraj=.2
        self.wtraj=.01
        self.theta_traj=0.0
        
        self.c12=.32
        self.g=19.2
        self.p_dist_h_1=0.0
        self.p_dist_h_2=0.0
        self.p_dist_nh=np.array([[-.09, 0.0],
                                 [0.0, -.09]])
        self.k_cap_dist=np.array([[-0.866],
                                  [-0.5]])
        self.ui_nh_dist = np.zeros((self.dim,1))
        self.ui_nh_dist=self.p_dist_nh.dot(self.k_cap_dist)
        

def main():
    my_robot = Robot()
    plt.ion()    
    sim_time=50
    for t in np.arange(0, sim_time,my_robot.T):
        my_robot.t1=my_robot.t1+.1
        my_robot.t_hist=np.append(my_robot.t_hist,my_robot.t1)

        my_robot.d_min_hist=np.append(my_robot.d_min_hist,my_robot.d_min)
        my_robot.d_max_hist=np.append(my_robot.d_max_hist,my_robot.d_max)
        
        if my_robot.demonstrated_traj:            
            f_vd=open("/home/sharma/Desktop/Final_ver_code_fc/vd.txt","r")
            lines_v=f_vd.readlines()
            my_robot.vtraj=float(lines_v[my_robot.vtext_line_counter])
            my_robot.vtext_line_counter=my_robot.vtext_line_counter+1
            
            f_wd=open("/home/sharma/Desktop/Final_ver_code_fc/wd.txt","r")
            lines_w=f_wd.readlines()
            my_robot.wtraj=float(lines_w[my_robot.wtext_line_counter])            
            my_robot.wtext_line_counter=my_robot.wtext_line_counter+1
            f_vd.close()
            f_wd.close()

        sat = 0.4
        my_robot.vtraj=np.clip(my_robot.vtraj, -sat, sat) 
        my_robot.wtraj=np.clip(my_robot.wtraj, -sat, sat) 

        my_robot.v_t[0,0]=my_robot.vtraj*np.cos(my_robot.theta_traj)-my_robot.wtraj*my_robot.l*np.sin(my_robot.theta_traj)
        my_robot.v_t[1,0]=my_robot.vtraj*np.sin(my_robot.theta_traj)+my_robot.wtraj*my_robot.l*np.cos(my_robot.theta_traj)
        my_robot.p_t[0,0]=my_robot.p_t[0,0]+my_robot.T*my_robot.v_t[0,0]
        my_robot.p_t[1,0]=my_robot.p_t[1,0]+my_robot.T*my_robot.v_t[1,0]
        my_robot.u_t[0,0]=(my_robot.v_t[0,0]-my_robot.v_t_prev[0,0])/my_robot.T
        my_robot.u_t[1,0]=(my_robot.v_t[1,0]-my_robot.v_t_prev[1,0])/my_robot.T
        my_robot.v_t_prev[0,0]=my_robot.v_t[0,0]
        my_robot.v_t_prev[1,0]=my_robot.v_t[1,0]
        my_robot.theta_traj=my_robot.theta_traj+my_robot.wtraj*my_robot.T
        

        my_robot.pt_x_hist=np.append(my_robot.pt_x_hist,my_robot.p_t[0,0])
        my_robot.pt_y_hist=np.append(my_robot.pt_y_hist,my_robot.p_t[1,0])
        my_robot.e=my_robot.p-my_robot.p_star
        
        my_robot.u1=my_robot.v_t-my_robot.v
        my_robot.u6=my_robot.p_t-my_robot.e
       
        my_robot.e1_x_hist=np.append(my_robot.e1_x_hist,my_robot.e[0,0])
        my_robot.e1_y_hist=np.append(my_robot.e1_y_hist,my_robot.e[1,0])

        my_robot.e2_x_hist=np.append(my_robot.e2_x_hist,my_robot.e[0,1])
        my_robot.e2_y_hist=np.append(my_robot.e2_y_hist,my_robot.e[1,1])

        my_robot.e3_x_hist=np.append(my_robot.e3_x_hist,my_robot.e[0,2])
        my_robot.e3_y_hist=np.append(my_robot.e3_y_hist,my_robot.e[1,2])
        

        my_robot.e_nh=my_robot.p_nh-my_robot.p
        my_robot.e_nh1_x_hist=np.append(my_robot.e_nh1_x_hist,my_robot.e_nh[0,0])
        my_robot.e_nh1_y_hist=np.append(my_robot.e_nh1_y_hist,my_robot.e_nh[1,0])

        my_robot.e_nh2_x_hist=np.append(my_robot.e_nh2_x_hist,my_robot.e_nh[0,1])
        my_robot.e_nh2_y_hist=np.append(my_robot.e_nh2_y_hist,my_robot.e_nh[1,1])

        my_robot.e_nh3_x_hist=np.append(my_robot.e_nh3_x_hist,my_robot.e_nh[0,2])
        my_robot.e_nh3_y_hist=np.append(my_robot.e_nh3_y_hist,my_robot.e_nh[1,2])        

        my_robot.Q_nh_1[0,0] = np.cos(my_robot.theta_nh[0])
        my_robot.Q_nh_1[0,1] = -my_robot.l*np.sin(my_robot.theta_nh[0])
        my_robot.Q_nh_1[1,0] = np.sin(my_robot.theta_nh[0])
        my_robot.Q_nh_1[1,1] = my_robot.l*np.cos(my_robot.theta_nh[0])

        my_robot.Q_nh_2[0,0] = np.cos(my_robot.theta_nh[1])
        my_robot.Q_nh_2[0,1] = -my_robot.l*np.sin(my_robot.theta_nh[1])
        my_robot.Q_nh_2[1,0] = np.sin(my_robot.theta_nh[1])
        my_robot.Q_nh_2[1,1] = my_robot.l*np.cos(my_robot.theta_nh[1])

        my_robot.Q_nh_3[0,0] = np.cos(my_robot.theta_nh[2])
        my_robot.Q_nh_3[0,1] = -my_robot.l*np.sin(my_robot.theta_nh[2])
        my_robot.Q_nh_3[1,0] = np.sin(my_robot.theta_nh[2])
        my_robot.Q_nh_3[1,1] = my_robot.l*np.cos(my_robot.theta_nh[2])        

        
        u_nh_1= np.linalg.inv(my_robot.Q_nh_1).dot(-my_robot.g*my_robot.e_nh[:,0]+my_robot.v[:,0]+my_robot.ui_nh_dist[:,0])
        u_nh_2= np.linalg.inv(my_robot.Q_nh_2).dot(-my_robot.g*my_robot.e_nh[:,1]+my_robot.v[:,1]+my_robot.ui_nh_dist[:,0])
        u_nh_3= np.linalg.inv(my_robot.Q_nh_3).dot(-my_robot.g*my_robot.e_nh[:,2]+my_robot.v[:,2]+my_robot.ui_nh_dist[:,0])

        
        my_robot.v_nh[:,0]=my_robot.Q_nh_1.dot(u_nh_1)
        my_robot.v_nh[:,1]=my_robot.Q_nh_2.dot(u_nh_2)
        my_robot.v_nh[:,2]=my_robot.Q_nh_3.dot(u_nh_3)
        

        
        my_robot.p_nh=my_robot.p_nh+my_robot.T*(my_robot.v_nh+.02*np.random.normal(0,1,1))

        my_robot.theta_nh[0]=my_robot.theta_nh[0]+my_robot.T*u_nh_1[1]
        my_robot.theta_nh[1]=my_robot.theta_nh[1]+my_robot.T*u_nh_2[1]
        my_robot.theta_nh[2]=my_robot.theta_nh[2]+my_robot.T*u_nh_3[1]

        my_robot.pa_x_nh_hist=np.append(my_robot.pa_x_nh_hist,my_robot.p_nh[0,0])
        my_robot.pa_y_nh_hist=np.append(my_robot.pa_y_nh_hist,my_robot.p_nh[1,0])
        my_robot.pb_x_nh_hist=np.append(my_robot.pb_x_nh_hist,my_robot.p_nh[0,1])
        my_robot.pb_y_nh_hist=np.append(my_robot.pb_y_nh_hist,my_robot.p_nh[1,1])
        my_robot.pc_x_nh_hist=np.append(my_robot.pc_x_nh_hist,my_robot.p_nh[0,2])
        my_robot.pc_y_nh_hist=np.append(my_robot.pc_y_nh_hist,my_robot.p_nh[1,2])
        

        my_robot.va_nh_hist=np.append(my_robot.va_nh_hist,u_nh_1[0])
        my_robot.vb_nh_hist=np.append(my_robot.vb_nh_hist,u_nh_2[0])
        my_robot.vc_nh_hist=np.append(my_robot.vc_nh_hist,u_nh_3[0])

        my_robot.wa_nh_hist=np.append(my_robot.wa_nh_hist,u_nh_1[1])
        my_robot.wb_nh_hist=np.append(my_robot.wb_nh_hist,u_nh_2[1])
        my_robot.wc_nh_hist=np.append(my_robot.wc_nh_hist,u_nh_3[1])
        
        for j in np.arange(0,my_robot.N):
            for k in np.arange(0,my_robot.N):                
                if (my_robot.a[j,k]==1):                    
                    my_robot.d_jk[j,k]=np.linalg.norm(my_robot.p[:,j]-my_robot.p[:,k])                    
                    if((np.linalg.norm(my_robot.p[:,j]-my_robot.p[:,k]))<=my_robot.R):
                        my_robot.u5[:,j]=my_robot.u5[:,j]+my_robot.a[j,k]*(my_robot.v[:,k]-my_robot.v[:,j])
                        my_robot.u4[:,j]=my_robot.u4[:,j]+my_robot.a[j,k]*(my_robot.e[:,k]-my_robot.e[:,j])
                        r_jk=np.linalg.norm(my_robot.p[:,j]-my_robot.p[:,k])
                        if(r_jk<=my_robot.r_out):
                            my_robot.u2[:,j]=my_robot.u2[:,j]+(r_jk*my_robot.Q_coll/(1+my_robot.Q_coll*(r_jk-my_robot.r_in)**2))*(my_robot.p[:,j]-my_robot.p[:,k])
                        else:
                            e_jk=np.linalg.norm(my_robot.e[:,j]-my_robot.e[:,k])
                            my_robot.u3[:,j]=my_robot.u3[:,j]+(e_jk*my_robot.Q_conn/(1+my_robot.Q_conn*(e_jk-my_robot.R)**2))*(my_robot.e[:,j]-my_robot.e[:,k])
                        d=np.linalg.norm(my_robot.p[:,j]-my_robot.p[:,k])
                        if(d<my_robot.d_min_hist[my_robot.t1]):
                            my_robot.d_min_hist[my_robot.t1]=d
                        if(d>my_robot.d_max_hist[my_robot.t1]):
                            my_robot.d_max_hist[my_robot.t1]=d
                        
                        my_robot.v_err[0,0] = my_robot.v_err[0,0] + (0.5*(my_robot.v[0,j] - my_robot.v[0,k])**2)
                        my_robot.v_err[1,0] = my_robot.v_err[1,0] + (0.5*(my_robot.v[1,j] - my_robot.v[1,k])**2)

                        my_robot.e_err[0,0] = my_robot.e_err[0,0] + (0.5*(my_robot.e[0,j] - my_robot.e[0,k])**2)
                        my_robot.e_err[1,0] = my_robot.e_err[1,0] + (0.5*(my_robot.e[1,j] - my_robot.e[1,k])**2)

            my_robot.u[0,j] = 2*my_robot.u1[0,j] + my_robot.u2[0,j] + my_robot.u3[0,j] + my_robot.u4[0,j] + my_robot.u5[0,j] + 4*my_robot.u6[0,j] + my_robot.u7[0,j]
            my_robot.u[1,j] = 2*my_robot.u1[1,j] + my_robot.u2[1,j] + my_robot.u3[1,j] + my_robot.u4[1,j] + my_robot.u5[1,j] + 4*my_robot.u6[1,j] + my_robot.u7[1,j]

            

            
            my_robot.p_dist_h_1 = -my_robot.c12*np.linalg.norm(-my_robot.u1[:,j])-.6
            my_robot.p_dist_h_2 = -my_robot.c12*np.linalg.norm(-my_robot.u1[:,j])-.6

            
            my_robot.v[0,j] = my_robot.v[0,j] + my_robot.T*(my_robot.u[0,j]+.01*np.random.normal(0,1,1)+my_robot.p_dist_h_1*my_robot.k_cap_dist[0])
            my_robot.v[1,j] = my_robot.v[1,j] + my_robot.T*(my_robot.u[1,j]+.01*np.random.normal(0,1,1)+my_robot.p_dist_h_2*my_robot.k_cap_dist[1])
  

            my_robot.p[0,j] = my_robot.p[0,j] + my_robot.v[0,j]*my_robot.T
            my_robot.p[1,j] = my_robot.p[1,j] + my_robot.v[1,j]*my_robot.T
        
        my_robot.d_12_hist=np.append(my_robot.d_12_hist,my_robot.d_jk[0,1])
        my_robot.d_13_hist=np.append(my_robot.d_13_hist,my_robot.d_jk[0,2])
        my_robot.d_23_hist=np.append(my_robot.d_23_hist,my_robot.d_jk[1,2])
        
        my_robot.e_err_x_hist=np.append(my_robot.e_err_x_hist,np.sqrt(my_robot.e_err[0,0]))
        my_robot.e_err_y_hist=np.append(my_robot.e_err_y_hist,np.sqrt(my_robot.e_err[1,0]))
        my_robot.v_err_x_hist=np.append(my_robot.v_err_x_hist,np.sqrt(my_robot.v_err[0,0]))
        my_robot.v_err_y_hist=np.append(my_robot.v_err_y_hist,np.sqrt(my_robot.v_err[1,0]))
        my_robot.e_err[0,0]=0.0
        my_robot.e_err[1,0]=0.0
        my_robot.v_err[0,0]=0.0
        my_robot.v_err[1,0]=0.0
        my_robot.pa_x_hist=np.append(my_robot.pa_x_hist,my_robot.p[0,0])
        my_robot.pa_y_hist=np.append(my_robot.pa_y_hist,my_robot.p[1,0])
        my_robot.pb_x_hist=np.append(my_robot.pb_x_hist,my_robot.p[0,1])
        my_robot.pb_y_hist=np.append(my_robot.pb_y_hist,my_robot.p[1,1])
        my_robot.pc_x_hist=np.append(my_robot.pc_x_hist,my_robot.p[0,2])
        my_robot.pc_y_hist=np.append(my_robot.pc_y_hist,my_robot.p[1,2])
        


    if my_robot.plot_flag==1:
        plt.rc('text', usetex=True)
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        pg.setConfigOptions(antialias=True)
        p1 = pg.plot(title="trajectory")
        p1.addLegend()
        p1.plot(my_robot.pt_x_hist, my_robot.pt_y_hist,  clear=False, pen=pg.mkPen('r', width=4), name='Traj*')

        p1.plot(my_robot.pa_x_hist, my_robot.pa_y_hist,  clear=False, pen=pg.mkPen('g', width=2), name='Traj_A')
        p1.plot(my_robot.pb_x_hist, my_robot.pb_y_hist,  clear=False, pen=pg.mkPen('b', width=2), name='Traj_B')
        p1.plot(my_robot.pc_x_hist, my_robot.pc_y_hist,  clear=False, pen=pg.mkPen('k', width=2), name='Traj_C')
        
        p1.plot(my_robot.pa_x_nh_hist, my_robot.pa_y_nh_hist,  clear=False, pen=pg.mkPen('g', width=4, style=QtCore.Qt.DashLine), name='Traj_A by nh')
        p1.plot(my_robot.pb_x_nh_hist, my_robot.pb_y_nh_hist,  clear=False, pen=pg.mkPen('b', width=4, style=QtCore.Qt.DashLine), name='Traj_B by nh')
        p1.plot(my_robot.pc_x_nh_hist, my_robot.pc_y_nh_hist,  clear=False, pen=pg.mkPen('k', width=4, style=QtCore.Qt.DashLine), name='Traj_C by nh')
        
        p1.plot(my_robot.pa_x_hist[100:101],my_robot.pa_y_hist[100:101], pen=None, symbol='o',name='A at t1')
        p1.plot(my_robot.pb_x_hist[100:101],my_robot.pb_y_hist[100:101], pen=None, symbol='o',name='B at t1')
        p1.plot(my_robot.pc_x_hist[100:101],my_robot.pc_y_hist[100:101], pen=None, symbol='o',name='C at t1')

        p1.plot(my_robot.pa_x_hist[150:151],my_robot.pa_y_hist[150:151], pen=None, symbol='s',name='A at t2')
        p1.plot(my_robot.pb_x_hist[150:151],my_robot.pb_y_hist[150:151], pen=None, symbol='s',name='B at t2')
        p1.plot(my_robot.pc_x_hist[150:151],my_robot.pc_y_hist[150:151], pen=None, symbol='s',name='C at t2')

        p1.plot(my_robot.pa_x_hist[200:201],my_robot.pa_y_hist[200:201], pen=None, symbol='o',name='A at t3')
        p1.plot(my_robot.pb_x_hist[200:201],my_robot.pb_y_hist[200:201], pen=None, symbol='o',name='B at t3')
        p1.plot(my_robot.pc_x_hist[200:201],my_robot.pc_y_hist[200:201], pen=None, symbol='o',name='C at t3')

        p1.plot(my_robot.pa_x_hist[300:301],my_robot.pa_y_hist[300:301], pen=None, symbol='s',name='A at t4')
        p1.plot(my_robot.pb_x_hist[300:301],my_robot.pb_y_hist[300:301], pen=None, symbol='s',name='B at t4')
        p1.plot(my_robot.pc_x_hist[300:301],my_robot.pc_y_hist[300:301], pen=None, symbol='s',name='C at t4')

        p1.setLabel('left', "y", units='m')
        p1.setLabel('bottom', "x", units='m')
       

        p3 = pg.plot(title="p_nh-ph")
        p3.addLegend()
        p3.plot(my_robot.t_hist, my_robot.e_nh1_x_hist,  clear=False, pen=pg.mkPen('k', width=2), name='A: px_nh-px_h')
        p3.plot(my_robot.t_hist, my_robot.e_nh1_y_hist,  clear=False, pen=pg.mkPen('k', width=4, style=QtCore.Qt.DashLine), name='A: py_nh-py_h')

        p3.plot(my_robot.t_hist, my_robot.e_nh2_x_hist,  clear=False, pen=pg.mkPen('r', width=2), name='B: px_nh-px_h')
        p3.plot(my_robot.t_hist, my_robot.e_nh2_y_hist,  clear=False, pen=pg.mkPen('r', width=4, style=QtCore.Qt.DashLine), name='B: py_nh-py_h')

        p3.plot(my_robot.t_hist, my_robot.e_nh3_x_hist,  clear=False, pen=pg.mkPen('b', width=2), name='C: px_nh-px_h')
        p3.plot(my_robot.t_hist, my_robot.e_nh3_y_hist,  clear=False, pen=pg.mkPen('b', width=4, style=QtCore.Qt.DashLine), name='C: py_nh-py_h')
        p3.setLabel('left', "Tracking_e", units='m')
        p3.setLabel('bottom', "Time", units='s')
              

        p8 = pg.plot(title="Distance")
        p8.addLegend()
        p8.plot(my_robot.t_hist, my_robot.d_12_hist,  clear=False, pen=pg.mkPen('r', width=2), name='d_12')
        p8.plot(my_robot.t_hist, my_robot.d_13_hist,  clear=False, pen=pg.mkPen('g', width=2), name='d_13')
        p8.plot(my_robot.t_hist, my_robot.d_23_hist,  clear=False, pen=pg.mkPen('b', width=2), name='d_23')

        p8.setLabel('left', "Distances", units='m')
        p8.setLabel('bottom', "Time", units='s')


        p9 = pg.plot(title="eix vs eiy*")
        p9.addLegend()
        p9.plot(my_robot.e1_x_hist, my_robot.e1_y_hist,  clear=False, pen=pg.mkPen('r', width=2), name='e1x vs e1y')
        p9.plot(my_robot.e2_x_hist, my_robot.e2_y_hist,  clear=False, pen=pg.mkPen('g', width=2), name='e2x vs e2y')
        p9.plot(my_robot.e3_x_hist, my_robot.e3_y_hist,  clear=False, pen=pg.mkPen('b', width=2), name='e3x vs e3y')

        p9.setLabel('left', "ey", units='m')
        p9.setLabel('bottom', "ey", units='m')

        p10 = pg.plot(title="v")
        p10.addLegend()
        p10.plot(my_robot.t_hist, my_robot.va_nh_hist,  clear=False, pen=pg.mkPen('r', width=2), name='va_nh')
        p10.plot(my_robot.t_hist, my_robot.vb_nh_hist,  clear=False, pen=pg.mkPen('g', width=2), name='vb_nh')
        p10.plot(my_robot.t_hist, my_robot.vc_nh_hist,  clear=False, pen=pg.mkPen('b', width=2), name='vc_nh')

        p10.setLabel('left', "Linear velocity", units='m/s')
        p10.setLabel('bottom', "Time", units='s')

        p11 = pg.plot(title="w")
        p11.addLegend()
        p11.plot(my_robot.t_hist, my_robot.wa_nh_hist,  clear=False, pen=pg.mkPen('r', width=2), name='wa_nh')
        p11.plot(my_robot.t_hist, my_robot.wb_nh_hist,  clear=False, pen=pg.mkPen('g', width=2), name='wb_nh')
        p11.plot(my_robot.t_hist, my_robot.wc_nh_hist,  clear=False, pen=pg.mkPen('b', width=2), name='wc_nh')

        p11.setLabel('left', "Angular velocity", units='rad/s')
        p11.setLabel('bottom', "Time", units='s') 
              

        pg.QtGui.QApplication.processEvents()
        raw_input("Press Enter to continue...")
if __name__ == '__main__':
    main()
