#MCMC v8 - placeholder code until I get around to neatening this up

import numpy as np
from numpy.polynomial import Polynomial
import matplotlib.pyplot as plot
from scipy.interpolate import make_interp_spline, BSpline
from scipy import optimize as scopt

#NOTE: using batches of random numbers is faster than one at a time
# implemented in the form of the 'random' arrays


class Lattice(object):

    def __init__(self,num,step,omega,path,ratio,action="harmonic",lam=50):
        self.N = num
        self.h = step
        if action == "anharmonic":
            self.f = omega
            self.w = 1
        else:
            self.w = omega
            self.f = 1
        self.acc_rate = ratio
        self.accept = 0
        self.type = action
        self.l = lam
        
        
        self.path = []
        if path == "cold":
            #cold start (all zeros)
            for i in range(0,num):
                self.path.append(1)
        elif path == "hot":
            #hot start (all random numbers)
            for i in range(0,num):
                self.path.append(2*np.random.random()-1)
        elif type(path) == array:
            self.path = path

    def action_harmonic_relax(self,v0,v1,v2):
        S = 0.5*((v2-v1)**2
             + (v1-v0)**2
             + self.w**2*v1**2)    
        return S

    def action_harmonic(self,v0,v1,v2):
        S = 0.5*((v2-v1)**2
             + self.w**2*v1**2)
        return S

    def action_anharmonic(self,v0,v1,v2):
        S = 0.5*((v2-v1)**2
             + (v1-v0)**2) + self.l*(v1**2 - self.f)**2
        return S


    def sweep(self,adjust_step=False):
        accepted = 0
            
        index = []
        for i in range(0,self.N):
            index.append(round(self.N*np.random.random()-0.5))

        random = [] 
        for i in range(0,2*self.N):
            random.append(np.random.random())

        for i in range(0,self.N):
            t = index[i]
            t_min = (t + self.N - 1) % self.N
            t_plus = (t + 1) % self.N
                
            x_new = self.path[t] + self.h*(random[i]-0.5)
            
            if self.type == "harmonic":
                S_old = self.action_harmonic_relax(self.path[t_min],self.path[t],self.path[t_plus])
                S_new = self.action_harmonic_relax(self.path[t_min],x_new,self.path[t_plus])
            elif self.type == "anharmonic":
                S_old = self.action_anharmonic(self.path[t_min],self.path[t],self.path[t_plus])
                S_new = self.action_anharmonic(self.path[t_min],x_new,self.path[t_plus])
            
            if random[self.N+i] < np.exp(S_old-S_new):
                self.path[t] = x_new
                accepted += 1

        #print(self.h,accepted,self.N,accepted/self.N)
        if adjust_step == True:
            if accepted == 0:
                self.h *= 0.1
            else:
                self.h *= accepted/(self.acc_rate*self.N)
        else:
            self.accept += accepted
                


class Simulation(object):
    
    def __init__(self):
        pass

    def set_lattice(self,size,stepsize,ratio,lattice_parameter,action="harmonic",lam=50):
        self.L = Lattice(size,stepsize,lattice_parameter,'cold',ratio,action,lam)

    def metropolis(self,num_sweeps,num_sep=1,therm=False,print_step=True,therm_step=True,graph_therm=False,therm_num=100):
        self.T = num_sweeps
        n = num_sep
        self.paths = []
        therm_paths = [list(self.L.path)]
        x2 = []
        if therm == True:
            for i in range(0,therm_num):
                if therm_step == True:
                    self.L.sweep(adjust_step=True)
                    if graph_therm == True:
                        therm_paths.append(list(self.L.path))
                        x2.append(np.array(self.L.path)**2)
                else:
                    self.L.sweep()
            if print_step == True:
                print("Thermalized step size:",self.L.h,"\n")
        self.paths.append(self.L.path)
        
        for i in range(0,self.T):
            for j in range(0,n):
                self.L.sweep()
            self.paths.append(list(self.L.path))

        self.all_x = Operators.paths_to_all(self.paths)

        if graph_therm == True:
            x2_avg = []
            for i in range(0,len(x2)):
                x2_avg.append(sum(x2[i])/len(x2[i]))
                

            fig = plot.figure()
            ax = plot.axes()
            ax.plot(therm_paths[0],"b",label="Initial")
            ax.plot(therm_paths[1],"r",label="1",linestyle="dashed")
            ax.plot(therm_paths[2],"g",label="2",linestyle="dotted")
            #ax.plot(therm_paths[3],"y",label="3",linestyle="dashdot")
            ax.plot(therm_paths[-1],"k",label="Final")
            ax.set_xlabel("Lattice time")
            ax.set_ylabel("x")
            ax.legend()

            fig2 = plot.figure()
            ax2 = plot.axes()
            ax2.plot(x2_avg,"k.")
            ax2.set_xlabel("Number of sweeps")
            ax2.set_ylabel("<x$^2$>")
            
            if self.L.type == "harmonic":
                R = 1 + (self.L.w**2)/2 - self.L.w*np.sqrt(1+(self.L.w**2)/4)
                R_term = ((1+R**self.L.N)/(1-R**self.L.N))
                x2 = (1/(2*self.L.w*np.sqrt(1+(self.L.w**2)/4)))*R_term
                ax2.hlines(x2,0,therm_num,"b")
            
            
            
            
        

    def get_values(self,text=True):
        #acceptance rate
        print("Acceptance rate = "+str(self.L.accept/(self.T*self.L.N))+"\n")
        
        #analytical values
        if self.L.type == "harmonic":
            R = 1 + (self.L.w**2)/2 - self.L.w*np.sqrt(1+(self.L.w**2)/4)
            R_term = ((1+R**self.L.N)/(1-R**self.L.N))
            x2 = (1/(2*self.L.w*np.sqrt(1+(self.L.w**2)/4)))*R_term
            x4 = 3*x2**2
            self.anal_x = [0,x2,0,x4]

        #expectation values
        self.exp_x = [0,0,0,0]
        for i in range(0,len(self.all_x)):
            for j in range(0,len(self.exp_x)):
                self.exp_x[j] += self.all_x[i]**(j+1)
        for i in range(0,len(self.exp_x)):
            self.exp_x[i] /= len(self.all_x)

        #unbiased error values
        self.err_x = [0,0,0,0]
        for i in range(0,len(self.exp_x)):
            for j in range(0,len(self.all_x)):
                self.err_x[i] += (self.all_x[j]**(i+1)-self.exp_x[i])**2/(len(self.all_x)-1)
            self.err_x[i] = np.sqrt(self.err_x[i]/len(self.all_x))

        if text == True:
            if self.L.type == "harmonic":
                print("Analytical values:")
                for i in range(0,len(self.anal_x)):
                    print("<x^"+str(i+1)+"> =",self.anal_x[i])
                
            print("\nExpectation values:")
            for i in range(0,len(self.exp_x)):
                print("<x^"+str(i+1)+"> =",self.exp_x[i])

            print("\nUnbiased error values:")
            for i in range(0,len(self.err_x)):
                print("Δx^"+str(i+1)+" =",self.err_x[i])
        

    def ac_errors(self,text=False,graph=False,fullgraph=False):
        path_avgs = [0,0,0,0]
        for i in range(0,len(path_avgs)):
            path_avgs[i] = Operators.x_average(Operators.x_power(self.paths,i+1))

        self.ac_err = [0,0,0,0]
        self.ac_factor = [0,0,0,0]
        for i in range(0,len(self.ac_err)):
            self.ac_err[i],self.ac_factor[i] = Operators.autocorrelation(path_avgs[i],graph)

        if text == True:
            print("\nAutocorrelation error values:")
            for i in range(0,len(self.ac_err)):
                print("Δx^"+str(i+1)+" =",self.ac_err[i])
            
        if fullgraph == True:
            Operators.autocorrelation_fullgraph(path_avgs[0])



    #for exp curve fitting
    def model(self,x,a,b):
        return a*np.exp(-b*x)

    def lattice_correlation(self,graph=False,print_text=False):
        G = []
        t = []
        dt = 0
        max_t = int(self.L.N/2)
        err_G = []

        check_E = []
        while dt < max_t:
            
            cor = []
            for j in range(0,len(self.paths)):
                 cor.append(Operators.correlation_x(self.paths[j],dt))
            g = sum(cor)/len(self.paths)

            if dt == 1:
                #print(cor)
                pass

            if dt > 0:
                check_E.append(-np.log(g/G[dt-1]))

            #if g < 0:
            #    dt = max_t
            
            if dt > 1:
                checker = check_E[dt-1]/check_E[dt-2]
            else:
                checker = 1
                
            if checker > 1.25 or checker < 0.75:
                dt = max_t
            #elif g < 0:
            #    dt = max_t
            else:
                G.append(g)
                t.append(dt)

                #errors
                '''
                sd = 0
                for i in range(0,len(cor)):
                    sd += (cor[i]-g)**2
                sd /= len(cor)-1
                err_G.append(np.sqrt(sd/(len(cor))))
                '''
                err_G_AC,f = Operators.autocorrelation(cor,False)
                err_G.append(err_G_AC)
                
                
            dt += 1

        #print(G)
        #print(err_G)

        #energy via curve fitting
        curve = scopt.curve_fit(self.model,t,G,sigma=err_G,absolute_sigma=True)
        
        new_t = np.linspace(min(t),max(t),100)
        new_G = self.model(new_t,curve[0][0],curve[0][1])

        self.E = curve[0][1]
        self.err_E = curve[1][1][1]

        #chi-squared values:
        chi_sq = 0
        for i in range(0,len(G)):
            exp_G = self.model(t[i],curve[0][0],curve[0][1])
            sd = err_G[i]**2*len(self.paths)
            #print(sd)
            chi_sq += (G[i]-exp_G)**2/sd
        chi_sq /= len(G)-2
            

        self.chi_sq = chi_sq

        if print_text == True:
            if self.L.type == "harmonic":
                print("\nContinuous E1-E0 = "+str(self.L.w))
                print("\nDiscrete E1-E0 = "+str(self.L.w*np.sqrt(1+self.L.w**2/4)))
                
            print("Sim E1-E0 = "+str(self.E)+" +- "+str(self.err_E))
            print("Reduced Chi-squared =",self.chi_sq)

        if graph == True:
            anal_t = np.linspace(min(t),max(t),100)
            anal_G = G[0]*np.exp(-self.L.w*np.sqrt(1+self.L.w**2/4)*anal_t)
        
            fig = plot.figure()
            ax = plot.axes()
            ax.set_xlabel("Δt")
            ax.set_ylabel("Correlation Function")
            #ax.set_title("Lattice Correlation")
            ax.set_yscale('log')
            ax.errorbar(t,G,yerr=err_G,fmt=".k")
            ax.plot(new_t,new_G,label="Model")
            ax.plot(anal_t,anal_G,label="Analytical")
            ax.legend()

            fig2 = plot.figure()
            ax2 = plot.axes()
            ax2.set_xlabel("Δt")
            ax2.set_ylabel("Correlation Function")
            #ax2.set_title("Lattice Correlation")
            ax2.errorbar(t,G,yerr=err_G,fmt=".k")
            ax2.plot(new_t,new_G,label="Model")
            ax2.plot(anal_t,anal_G,label="Analytical")
            ax2.legend()

    def lattice_correlation_fullgraphs(self):
        G = []
        t = []
        dt = 0
        max_t = self.L.N
        err_G = []
        while dt < max_t:
            
            cor = []
            for j in range(0,len(self.paths)):
                 cor.append(Operators.correlation_x(self.paths[j],dt))
            g = sum(cor)/len(self.paths)

            G.append(g)
            t.append(dt)

            #errors
            err_G_AC,f = Operators.autocorrelation(cor,False)
            err_G.append(err_G_AC)
                
            dt += 1

        #y_err = np.abs(np.log(np.array(err_g)))
        #y_err2 = np.abs(np.array(err_g)/np.array(G))
            
        #curve fitting
        #curve = scopt.curve_fit(self.model_G,t,G,sigma=err_G,absolute_sigma=True)
        #print(curve)
        
        #new_t = np.linspace(min(t),max(t),100)
        #new_G = self.model_G(new_t,curve[0][0],curve[0][1],curve[0][2],curve[0][3])

        #print("E =",curve[0][1])
        #print("A =",curve[0][0])

        fig = plot.figure()
        ax = plot.axes()
        ax.errorbar(t,G,yerr=err_G,fmt="k.")
        ax.plot(t,G,"r")
        ax.set_ylabel("Average G")
        ax.set_xlabel("Δt")
        ax.set_yscale('log')

        #anal_t = np.linspace(min(t),max(t),100)
        #anal_G = G[0]*np.exp(-self.L.w*anal_t)
        #ax.plot(anal_t,anal_G)

        fig = plot.figure()
        ax2 = plot.axes()
        ax2.errorbar(t,G,yerr=err_G,fmt="k.")
        #ax2.plot(new_t,new_G,"r",label="Model")
        ax2.set_ylabel("Average G")
        ax2.set_xlabel("Δt")


    def model_G(self,x,a,b,c,d):
        return b*np.cosh(a*(x-d)) + c*(x-d)**2
        


    def create_pdf_x(self,num_points):
        xmax = max(self.all_x)
        xmin = min(self.all_x)

        binsize = (xmax-xmin)/num_points
    
        xold = np.linspace(xmin,xmax,num_points)

        xnew = np.linspace(xmin,xmax,num_points)

        fig = plot.figure()
        ax = plot.axes()
    
        ax.hist(self.all_x,num_points,label="Model",density=True)
        
        if self.L.type =="harmonic":
            pdf = np.sqrt(self.L.w*(1+self.L.w**2/4)/np.pi)*np.exp(-self.L.w*(1+self.L.w**2/4)*xnew**2)
            continuum = np.sqrt(self.L.w/np.pi)*np.exp(-self.L.w*xnew**2)
            
            ax.plot(xnew,pdf,label="Analytical",color="black")
            ax.plot(xnew,continuum,label="Continuum Limit",color="r")
            ax.vlines(0,0,np.sqrt(self.L.w/np.pi),color="black")
            
        ax.set_xlabel("x")
        ax.set_ylabel("|Ψ$_0$|$^2$")
        ax.legend()
        ax.vlines(self.exp_x[0],0,np.sqrt(self.L.w/np.pi),linestyle="dotted",color="orange")

    def anharmonic_energy(self):
        self.E0 = self.L.l*(3*self.exp_x[3] - 4*self.L.f*self.exp_x[1] + self.L.f**2)
        print("E0 =",self.E0)


class Operators(object):
    def x_average(configs):
        assert len(configs)>0 and len(configs[0])>0
        path_avgs = []
        for i in range(0,len(configs)):
            path_avgs.append(0)
            for j in range(0,len(configs[i])):
                path_avgs[i] += configs[i][j]
            path_avgs[i] /= len(configs[i])
        return path_avgs
            

    def x_power(configs,power):
        new_configs = []
        for i in range(0,len(configs)):
            new_configs.append([])
            for j in range(0,len(configs[i])):
                new_configs[i].append(configs[i][j]**power)
        return new_configs


    def paths_to_all(configs):
        all_values = []
        for i in range(0,len(configs)):
            for j in range(0,len(configs[i])):
                all_values.append(configs[i][j])
        return all_values
                

    def correlation_x(path,t):
        N = len(path)

        g = 0
        for i in range(0,N):
            g += path[i]*path[(i+t)%N]
        g /= N

        return g


    def autocorrelation(data,graph=False):
        N = len(data)
        avg = sum(data)/N
        A0 = 0
        for i in range(0,N):
            A0 += (data[i]-avg)**2
        A0 /= N-1

        if A0 == 0:
            print("HELP!")
        
        A_lag = [1]

        max_lag = N-1
        lag = 1
        while lag < max_lag:
            E = 0
            for i in range(0,N-lag):
                E += (data[i]-avg)*(data[i+lag]-avg)
            E /= (N-lag)

            if E < 0:
                lag = max_lag
            else:
                A_lag.append(E/A0)

            lag += 1

        A = 1+2*sum(A_lag)

        auto_err = np.sqrt(A0*A/N)
      
        if graph == True:
            fig = plot.figure()
            ax = plot.axes()
            ax.set_ylabel("Normalised Autocorrelation Function")
            ax.set_xlabel("Lag")
            ax.plot(A_lag)

        return auto_err,A


    def autocorrelation_fullgraph(data):
        N = len(data)
        avg = sum(data)/N
        A0 = 0
        for i in range(0,N):
            A0 += (data[i]-avg)**2
        A0 /= N-1
        
        A_lag = []
        integral = [0]
        cut_off = False

        max_lag = N-1
        lag = 0
        while lag < max_lag:
            E = 0
            for i in range(0,N-lag):
                E += (data[i]-avg)*(data[i+lag]-avg)
            E /= (N-lag)

            if E > 0 and cut_off == False:
                integral.append(E/A0)
            else:
                cut_off = True
            
            A_lag.append(E/A0)

            lag += 1
      
        fig = plot.figure()
        ax = plot.axes()
        ax.set_ylabel("Normalized Autocorrelation Function")
        ax.set_xlabel("Lag")
        ax.plot(A_lag,"k")
        ax.fill(integral,"r")
        ax.text(70,0.95,"Area = Integrated Autocorrelation Function")



class Test(object):
    def __init__(self):
        pass

    def harmonic(self):
        
        sim = Simulation()

        sim.set_lattice(size=100,stepsize=1,ratio=0.5,lattice_parameter=0.001)

        sim.metropolis(num_sweeps=10000,num_sep=1,therm=True,print_step=True,graph_therm=False,therm_num=1000)

        sim.get_values()

        #parameters: text,graph,fullgraph)
        sim.ac_errors(True,False,False)

        sim.lattice_correlation(True,True)
        sim.lattice_correlation_fullgraphs()

        sim.create_pdf_x(50)

        x2_avg = []
        x2_paths = Operators.x_power(sim.paths,2)
        for i in range(0,len(x2_paths)):
            x2_avg.append(sum(x2_paths[i])/len(x2_paths[i]))

        '''
        fig2 = plot.figure()
        ax2 = plot.axes()
        ax2.plot(x2_avg,"k.")
        ax2.set_xlabel("Number of sweeps")
        ax2.set_ylabel("<x$^2$>")
        '''

        #fig2 = plot.figure()
        #ax2 = plot.axes()
        #ax2.plot(sim.all_x)

        plot.show()

    def anharmonic(self):

        sim = Simulation()

        sim.set_lattice(size=50,stepsize=1,ratio=0.5,lattice_parameter=1,action="anharmonic",lam=2)

        sim.metropolis(num_sweeps=10000,num_sep=1,therm=True,print_step=True,graph_therm=True,therm_num=1000)

        sim.get_values()
        sim.ac_errors(True,False,True)

        sim.anharmonic_energy()
        sim.lattice_correlation(True,True)

        sim.create_pdf_x(50)

        plot.show()


test = Test()

#test.harmonic()
test.anharmonic()



