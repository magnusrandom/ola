# Import Data

from dateutil.parser import parse 
import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams
from numpy import sin, linspace, sqrt, diag
from pylab import *

df = pd.read_csv('https://github.com/selva86/datasets/raw/master/AirPassengers.csv')

# Draw Plot
#plt.figure(figsize=(12,6), dpi= 80)
rcParams['figure.figsize'] = 12, 6


#plt.plot('date', 'traffic', data=df, color='tab:red')

#plt.style.use('default')  # coloquei o tema do plot como padrao 
rcParams['figure.figsize'] = 12, 6 #tamanho da figura



plt.rcParams['font.family'] = 'Palatino'
#plt.rcParams['font.serif'] = 'Sans'
#plt.rcParams['font.monospace'] = 'Ubuntu Mono'
#rc('font',**{'family':'serif','serif':['Sans']


SMALL_SIZE = 14  # 14 before
MEDIUM_SIZE = 18 # 18 BEFORE
BIGGER_SIZE =  18 #it was 18 before

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  #









data = pd.read_csv('resistividade.txt', sep='\s+',header=None,comment='@')

data = pd.DataFrame(data)  # vai ler o arquivo usando panda no formato Data frame
x = data[0] # escolhi a primeira coluna como x
y = data[1] # segunda coluna como y


plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{siunitx}')






def r_squared(x, y, popt, f):
    res = y - f(x, *popt)
    ss_r = sum(res**2)
    ss_t = sum((y - mean(y))**2)

    r2 = 1 - (ss_r / ss_t)
    return r2




def f(x, a, b, c):
    y =  a*x**2+ b + c*x 
    return y

# we'd like to know what a, b and c are.
# thankfully, we can use SciPy
from scipy import optimize


popt, pcov = optimize.curve_fit(f, x, y)
a, b, c = popt
sa, sb, sc = sqrt(diag(pcov))
print(f"a = {a}, b = {b}, c = {c}")
print(f"σa = {sa}, σb = {sb}, σc = {sc}")



R2 = r_squared(x, y, popt, f)
print(f"R² = {R2}")
xs = linspace(0.05, 0.6, 512)
ys = f(xs, a, b, c)






















# Decoration
plt.ylim(0.1, 18)
plt.xlim(0, 0.65)
#xtick_location = df.index.tolist()[::12]
#xtick_labels = [x[-4:] for x in df.date.tolist()[::12]]
#plt.xticks(ticks=xtick_location, labels=xtick_labels, rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
#plt.yticks(fontsize=12, alpha=.7)
#plt.title("Air Passengers Traffic (1949 - 1969)", fontsize=22)
#plt.grid(axis='both', alpha=.3)
#dfc588
plt.plot(x, y, color="darkred",marker='.',linestyle='',linewidth=2, markersize=6)
plot(xs, ys, color='#17100e',linewidth=2,label="reta da regressão linear")
# Remove borders
plt.gca().spines["top"].set_alpha(0.0)    
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.0)    
plt.gca().spines["left"].set_alpha(0.3)   
#plt.gca().xaxis.set_ticklabels([])
plt.gca().xaxis.set_ticks_position('none') 
plt.gca().yaxis.set_ticks_position('none') 

plt.ylabel(r' Resistência\,($ \Omega $)')
plt.xlabel(r' Distância\,(m) ')
plt.title(r"Resistência versus distância ao longo de um fio")

#plt.savefig("rVl3.png", dpi=600, bbox_inches='tight')

legend = plt.legend(loc=4)

legend.get_frame().set_facecolor('none')


legend.get_frame().set_linewidth(0.0)
plt.savefig("rVl4.png", dpi=600, bbox_inches='tight')

#plt.legend(loc=4)
plt.show()
