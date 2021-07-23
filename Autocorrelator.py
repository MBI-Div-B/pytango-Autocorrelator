#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 16:03:29 2021

@author: moke
"""

from tango import AttrWriteType, DevState, DevFloat, DebugIt, AttributeProxy
from tango.server import Device, attribute, device_property
from scipy.constants import c as co
import tango
import lmfit as lm
import numpy as np
# from math import cos, sin

class Autocorrelator(Device):
    '''
    Autocorrelator
    
    This device can calculate the pulsewidth in fs of the incoming laser 
    detected by the selected camera.
    
    To select which is the camera connected to the autocorrelator, one must write 
    in dev_properties from the Basler device in Jive, the serial number of 
    the desired camera, which is usually written on the camera as S/N).
    
    Then this code will automatically receive a matrix with all the values
    (the image itself) and one will be able to calculate the FWHM in x and y 
    under different situations.
    '''   

    ImageProxy = device_property(
        dtype=str,
        default_value='domain/family/member/attribute',
        )
    
    data_x = attribute(name='data_x', label='data x',max_dim_x=4096,
                      dtype=(DevFloat,), access=AttrWriteType.READ)
    data_y = attribute(name='data_y', label='data y',max_dim_x=4096,
                      dtype=(DevFloat,), access=AttrWriteType.READ)
    fitting_x = attribute(name='fitting_x', label='fitting x',max_dim_x=4096,
                      dtype=(DevFloat,), access=AttrWriteType.READ)
    fitting_y = attribute(name='fitting_y', label='fitting y',max_dim_x=4096,
                      dtype=(DevFloat,), access=AttrWriteType.READ)
    mu_x = attribute(name='mu_x', label='Location of maximum in data x (pixels)', 
                          dtype="float", format="%4.3f", access=AttrWriteType.READ)
    
    center1 = attribute(label='peak 1, without the quartz (loc. max in pixels)', dtype="float", 
                        access=AttrWriteType.READ_WRITE, memorized=True, hw_memorized=True)
    
    center2 = attribute(label='peak 2, with the quartz (loc. max in pixels)', dtype="float", 
                        access=AttrWriteType.READ_WRITE, memorized=True, hw_memorized=True)
    
    calibration = attribute(name='calibration', label='calibration (fs/pixel)', 
                          dtype="float",format="%4.3f", access=AttrWriteType.READ)
    
    gaussian_pulsewidth_x = attribute(name='gaussian_pulsewidth_x', label='gauss pulsewidth: x axis(fs)', 
                            dtype="float",format="%4.3f", access=AttrWriteType.READ)


    gaussian_pulsewidth_y = attribute(name='gaussian_pulsewidth_y', label='y axis(fs)', 
                            dtype="float",format="%4.3f", access=AttrWriteType.READ)
    sech2_pulsewidth_x = attribute(name='sech2_pulsewidth_x', label='sech2 pulsewidth: x axis (fs)', 
                            dtype="float",format="%4.3f", access=AttrWriteType.READ)
    sech2_pulsewidth_y = attribute(name='sech2_pulsewidth_y', label='y axis (fs)', 
                            dtype="float",format="%4.3f", access=AttrWriteType.READ)

    # twodgaussian = attribute(name='twodgaussian', label='two dimensional integration', 
    #                       dtype="float", format="%4.3f", access=AttrWriteType.READ)
    
    sigma_try = device_property(dtype="int")
    __position_1 = 780
    __position_2 = 1129
    d = 95*1e-6 #conversion into m
    c_0 = co*1e-15  #conversion into m/fs
    width = 300
    sqrt2 = np.sqrt(2)
    N = 0
    def init_device(self):
        self.debug_stream("Preparing device")
        Device.init_device(self)
        try:
            self.image_proxy = AttributeProxy(self.ImageProxy)
            self.debug_stream('Init was done')
        except:
            self.error_stream('Could not contact camera :( ')
            self.set_state(DevState.OFF)
        self.set_state(DevState.ON)
            
    def read_data_x(self):
        self.debug_stream("Graphing x axis")
        real_data = np.array(self.image_proxy.read().value)
        self.N = len(real_data[0,:])
        self.x2 = np.linspace(0,self.N,self.N)
        self.x_axis = np.mean(real_data, axis = 0)
        self.debug_stream('cameras x axis was graphed colected properly')
        return self.x_axis
    
    def read_data_y(self):
        self.debug_stream("Graphing y axis")
        real_data = np.array(self.image_proxy.read().value)
        self.N2 = len(real_data[:,0])

        self.y2 = np.linspace(0,self.N2,self.N2)
        self.y_axis = np.mean(real_data, axis = 1)
        self.debug_stream('cameras y axis was graphed properly')
        return self.y_axis
        
    def read_mu_x(self):
        self.debug_stream("Calculating the center of the peak in the x axis")
        def gaussian_paula(x, mu, A, sigma, c):
            return A*np.exp(-(x-mu)**2/(2*sigma**2))+c

        mod = lm.Model(gaussian_paula)
        self.parsx = lm.Parameters()
        real_data = np.array(self.image_proxy.read().value)
        self.x_axis = np.mean(real_data, axis = 0)
        
        self.x_max = np.max(self.x_axis)
        self.x_min = np.min(self.x_axis)
        self.mu = self.x_axis.argmax()
        
        self.parsx.add('mu', value = self.mu)
        self.parsx.add('A', value = self.x_max-self.x_min)
        self.parsx.add('c', value = self.x_min)
        self.parsx.add('sigma', value = self.sigma_try )
        
        self.debug_stream('parameters fitting x axis data')
        self.out_gaussx = mod.fit(self.x_axis, self.parsx, x=self.x2) 
        self.debug_stream('fitting x axis done succesfully') 
        
        self.a = self.out_gaussx.best_values['sigma']
        return self.out_gaussx.best_values['mu']
    
    def read_fitting_x(self):
        return self.out_gaussx.best_fit
    
    def read_center1(self):
        return self.__position_1
    
    def write_center1(self, value):
        self.__position_1 = value
        
    def read_center2(self):
        return self.__position_2
    
    def write_center2(self, value):
        self.__position_2 = value
        
    def read_calibration(self):
        t = self.d/self.c_0
        self.calibr_real = abs(t/(self.__position_1 - self.__position_2))
        return self.calibr_real

    def read_gaussian_pulsewidth_x(self):
        return self.a*2*np.sqrt(2*np.log(2))*self.calibr_real*self.sqrt2
    
    
    def read_sech2_pulsewidth_x(self):
        return self.a*2*np.sqrt(2*np.log(2))*self.calibr_real*1.55  
    
    def read_gaussian_pulsewidth_y(self):
        def gaussian_paula(x, mu, A, sigma, c):
            return A*np.exp(-(x-mu)**2/(2*sigma**2))+c
        
        mod = lm.Model(gaussian_paula)
        self.pars = lm.Parameters()
        
        real_data = self.image_proxy.read().value
        self.y_axis = np.mean(real_data, axis = 1)
        
        self.y_max = np.max(self.y_axis)
        self.y_min = np.min(self.y_axis)
        self.muu = self.y_axis.argmax()
        
        self.pars.add('mu', value = self.muu)
        self.pars.add('A', value = self.y_max-self.y_min)
        self.pars.add('c', value = self.y_min)
        self.pars.add('sigma', value = self.sigma_try )
        
        self.debug_stream('parameters fitting y axis data') 
        
        self.out_gaussy = mod.fit(self.y_axis, self.pars, x=self.y2) 
        self.b = self.out_gaussy.best_values['sigma']        
        self.debug_stream('fitting y axis done succesfully')
        return self.b*2*np.sqrt(2*np.log(2))*self.calibr_real*self.sqrt2
    
    def read_sech2_pulsewidth_y(self):
        return self.b*2*np.sqrt(2*np.log(2))*self.calibr_real*1.55

    def read_fitting_y(self):
        return self.out_gaussy.best_fit


'''this is what I tried to fit to a 2D data but its too heavy. however, it may be
useful when the pulse is tilted in one direction diferent than x or y'''
    # def read_twodgaussian(self):
    #     def gaussian_paula(x, mux, muy, A, sigma_x, sigma_y, c, theta):
    #         print('Calling gaussian')
    #         a = ((cos(theta)**2/sigma_x**2)+(sin(theta)**2/sigma_y**2))
    #         b = (-(sin(2*theta)/sigma_x**2)+(sin(2*theta)/sigma_y**2))
    #         c = ((sin(theta)**2/sigma_x**2)+(cos(theta)**2/sigma_y**2))
    #         u = x[:, 0]
    #         v = x[:, 1]
    #         gauss = 0.5*A*np.exp(-(a*(u-mux)**2+b*(u-mux)*(v-muy)+c*(v-muy)**2))+c
    #         return gauss

    #     self.camera_data = self.camera.image
    #     self.camera_data = np.delete(self.camera_data, np.s_[::2], axis=1)
    #     self.camera_data = np.delete(self.camera_data, np.s_[::2], axis=0)
    #     self.camera_data = np.delete(self.camera_data, np.s_[::2], axis=1)
    #     self.camera_data = np.delete(self.camera_data, np.s_[::2], axis=0)
      
    #     self.N = len(self.camera_data[:,0])

    #     self.N22 = self.N*self.N
    #     self.N2 = len(self.camera_data[0])

        # self.margin = int(abs((self.N-self.N2)/2))

        # self.real_data = self.camera_data[:,self.margin:-self.margin]


        # self.x_axis = np.mean(self.real_data, axis = 0)
        # self.y_axis = np.mean(self.real_data,axis = 1)         
        # self.x2 = np.linspace(0,self.N,self.N)
        # self.debug_stream('data colected')
        
        # self.mod = lm.Model(gaussian_paula)
        # self.pars = lm.Parameters()
        # self.x_max = np.max(self.x_axis)
        
        # self.x_min = np.min(self.x_axis)
        # self.mux = self.x_axis.argmax()
        # self.in_theta = 0
        
        # self.y_max = np.max(self.y_axis)
        # self.y_min = np.min(self.y_axis)
        # self.muy = self.y_axis.argmax() 

        # self.amplitude = (self.x_max-self.x_min)/2+(self.y_max-self.y_min)/2

        # self.offset = (self.x_min+self.y_min)/2

        # vary = True

        # self.pars.add('theta', value = self.in_theta, vary = vary)
        # self.pars.add('sigma_x', value = self.sigma_try, vary = vary)
        # self.pars.add('sigma_y', value = self.sigma_try, vary = vary)
        # self.pars.add('c', value = self.offset, vary = vary)
        # self.pars.add('A', value = self.amplitude, vary = vary)
        
        # self.pars.add('mux', value = self.mux, vary = vary)
        # self.pars.add('muy', value = self.muy, vary = vary)
        
        
        # self.debug_stream('guess params added')
        # self.data = np.zeros((self.N22,3))
        # self.x, self.y = np.meshgrid(self.x2, self.x2)
        
        # self.data[:,0] = self.x.flatten()
        # self.data[:,1] = self.y.flatten()
        # self.data[:,2] = self.real_data.flatten()
        # self.debug_stream('before the bug')        
        
        # self.out_gauss = self.mod.fit(self.data[:,2],  x=self.data[:, 0:2], params=self.pars)
        # self.debug_stream('no more bug')
        # return self.out_gauss.best_values['mux']
        
if __name__ == "__main__":
    Autocorrelator.run_server()



