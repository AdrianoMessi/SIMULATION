import numpy as np
import math
import random
import matplotlib.pyplot as plt
import time
from scipy import stats
import sympy as sp
from scipy.stats import norm
import pandas as pd
from datetime import datetime
from scipy.stats import poisson
from scipy.stats import uniform

class Congruencial_multiplicativo:
    def __init__(self, x0, b, m):
        self.x0 = x0
        self.b = b
        self.m = m
        self.x00 = x0
        
    def random(self):
        xi = (self.x0*self.b) % self.m
        self.x0 = xi 
        return xi/self.m
    
    def randoms(self):
        nums = []
        xi_anterior = self.x00
        for i in range(self.m - 1):
            xi = (xi_anterior*self.b) % self.m
            if xi/self.m in nums: break
            else: 
                nums.append(xi/self.m)
                xi_anterior = xi   # actualizando la x(i-1) 
        return nums

    
class Congruencial_aditivo:
    def __init__(self, x0, b, m):
        self.x0 = x0
        self.b = b
        self.m = m
        self.x00 = x0
        
    def random(self):
        xi = (self.x0 + self.b) % self.m
        self.x0 = xi 
        return xi/self.m
    
    def randoms(self):
        nums = []
        xi_anterior = self.x00
        for i in range(self.m - 1):
            xi = (xi_anterior + self.b) % self.m
            if xi/self.m in nums: break
            else: 
                nums.append(xi/self.m)
                xi_anterior = xi   # actualizando la x(i-1) 
        return nums
    

class Congruencial_polinomial:
    def __init__(self, x0, b_s, m):
        self.x0 = x0
        self.b_s = np.array(b_s)
        self.m = m
        self.grado = len(b_s) - 1
        self.x00 = x0
        
    def random(self):
        x_s = np.array([self.x0**i for i in range(self.grado + 1)])
        xi = (np.dot(x_s, self.b_s)) % self.m
        self.x0 = xi 
        return xi/self.m
    
    def randoms(self):    
        nums = []
        xi_anterior = self.x00
        
        for i in range(self.m - 1):
            x_s = np.array([xi_anterior**i for i in range(self.grado + 1)])           
            xi = (np.dot(x_s, self.b_s)) % self.m
            if xi/self.m in nums: break
            else: 
                nums.append(xi/self.m)
                xi_anterior = xi   # actualizando la x(i-1) 

                
        return nums
    
    
class Congruencial_multiplicativo_multiple:
    def __init__(self, x0_s, b_s, m):
        self.x0_s = np.array(x0_s)
        self.b_s = np.array(b_s)
        self.m = m
        # self.xi = 1
        
    def random(self):
        xi = np.dot(self.x0_s, self.b_s) % self.m
        
        self.x0_s = np.delete(self.x0_s, -1)
        self.x0_s = np.concatenate((np.array([xi]), self.x0_s))
        return xi/self.m
    
    def randoms(self, N):
        pseudos = []
        for i in range(N): 
            num = Congruencial_multiplicativo_multiple.random(self)
            if num in pseudos:
                print(f'Con los datos proporcionados, solo se pudo obtener una muestra de {i} numeros pseudoaleatorios')
                break
            else: 
                pseudos.append(num)
        return pseudos

    
class Congruencial_combinado:
    def __init__(self, x0, y0, b1, b2, m1, m2):
        self.x0 = x0
        self.y0 = y0
        self.b1 = b1
        self.b2 = b2
        self.m1 = m1
        self.m2 = m2
        self.x00 = x0
        self.y00 = y0
        
        
    def random(self):
        xi = (self.x0*self.b1) % self.m1
        yi = (self.y0*self.b2) % self.m2
        zi = (xi - yi) % self.m1
        
        self.x0 = xi
        self.y0 = yi
        
        return zi/self.m1
    
    def randoms(self):
        nums = []
        xi_anterior = self.x00
        yi_anterior = self.y00
        for i in range(self.m1 - 1):
            xi = (xi_anterior*self.b1) % self.m1
            yi = (yi_anterior*self.b2) % self.m2
            zi = (xi - yi) % self.m1
            
            if zi/self.m1 in nums: break
            else: 
                nums.append(zi/self.m1)
                xi_anterior = xi   # actualizando la x(i-1)
                yi_anterior = yi   # actualizando la y(i-1)
                
        return nums

    
    
class Congruencial_combinado_multiple:
    def __init__(self, x0_s, y0_s, b1_s, b2_s, m1, m2):
        self.x0_s = np.array(x0_s)
        self.y0_s = np.array(y0_s)
        self.b1_s = np.array(b1_s)
        self.b2_s = np.array(b2_s)
        self.m1 = m1
        self.m2 = m2
        
        
    def random(self):
        xi = np.dot(self.x0_s, self.b1_s) % self.m1
        yi = np.dot(self.y0_s, self.b2_s) % self.m2
        zi = (xi - yi) % self.m1

        # actualizamos a las xi, yi
        self.x0_s = np.delete(self.x0_s, -1)
        self.x0_s = np.concatenate((np.array([xi]), self.x0_s))
        self.y0_s = np.delete(self.y0_s, -1)
        self.y0_s = np.concatenate((np.array([yi]), self.y0_s))
        return zi/self.m1

    
    def randoms(self, N):
            pseudos = []
            for i in range(N): 
                num = Congruencial_combinado_multiple.random(self)
                if num in pseudos:
                    print(f'Con los datos proporcionados, solo se pudo obtener una muestra de {i} numeros pseudoaleatorios')
                    print(num)
                    break
                else: 
                    pseudos.append(num)
            return pseudos
    
    
class Distribuciones:
    def masabinom(n, p, x):
        return math.comb(n, x) * (p**x) * (1-p)**(n-x)

    def masahyper(N, k, n, x):
        return math.comb(k, x)*math.comb(N-k, n-x) / (math.comb(N, n))

    def masageom(p, x):
        return p * (1-p)**(x)

    def masanbinom(r, p, x):
        return math.comb(x + r - 1, x) * (1-p)**x * p**r
    

    
class Numgen:
    def unif(a, b, n):
        muestra = []
        for i in range(n):
            lista = [x for x in range(a, b)]
            muestra.append(random.choice(lista))
        return muestra
        
    def unif_arbitrario(lista, n):
        muestra = []
        for i in range(n):
            muestra.append(random.choice(lista))
        return muestra

    def bernoulli(p, n):
        muestra = []
        for _ in range(n):
            num = random.random()
            if num < p:
                muestra.append(1)
            elif num > p:
                muestra.append(0)
                
        return muestra
        
    def binomial(n, p, N):
        soporte = []
        for i in range(n+1):
            soporte.append(i)
        probas = []
        for valor in soporte:
            probas.append(Distribuciones.masabinom(n, p, valor))
        lista_combinada = list(zip(probas, soporte))
        lista_combinada.sort(key=lambda x: x[0])
        probas1, soporte1 = zip(*lista_combinada)
    
        probas_acum = np.cumsum(probas1)
        muestra = []
        for _ in range(N):
            num = random.random()
            for i in range(len(probas1)):
                if num < probas_acum[i]:
                    muestra.append(soporte1[i])
                    break
    
                elif i == len(probas1)-1:
                    muestra.append(soporte1[i])
                    break
        return muestra

    def geom(p, n):
        soporte = []
        for i in range(100):
            soporte.append(i)

        probas = []
        for valor in soporte:
            probas.append(Distribuciones.masageom(p, valor))
        lista_combinada = list(zip(probas, soporte))
        lista_combinada.sort(key=lambda x: x[0])
        probas1, soporte1 = zip(*lista_combinada)

        probas_acum = np.cumsum(probas1)
        muestra = []
        for _ in range(n):
            num = random.random()
            for i in range(len(probas1)):
                if num < probas_acum[i]:
                    muestra.append(soporte1[i])
                    break
    
                elif i == len(probas1)-1:
                    muestra.append(soporte1[i])
                    break
        return muestra
        

    def nbinomial(r, p, n):
        soporte = []
        for i in range(100):
            soporte.append(i)

        probas = []
        for valor in soporte:
            probas.append(Distribuciones.masanbinom(r, p, valor))
        lista_combinada = list(zip(probas, soporte))
        lista_combinada.sort(key=lambda x: x[0])
        probas1, soporte1 = zip(*lista_combinada)

        probas_acum = np.cumsum(probas1)
        muestra = []
        for _ in range(n):
            num = random.random()
            for i in range(len(probas1)):
                if num < probas_acum[i]:
                    muestra.append(soporte1[i])
                    break
    
                elif i == len(probas1)-1:
                    muestra.append(soporte1[i])
                    break
        return muestra

    def hyper(N, k, n, m):    
        soporte = []
        for i in range(k+1):
            if i > n:
                break
            else:
                soporte.append(i)

        probas = []
        for valor in soporte:
            probas.append(Distribuciones.masahyper(N, k, n, valor))
        lista_combinada = list(zip(probas, soporte))
        lista_combinada.sort(key=lambda x: x[0])
        probas1, soporte1 = zip(*lista_combinada)

        probas_acum = np.cumsum(probas1)
        muestra = []
        for _ in range(m):
            num = random.random()
            for i in range(len(probas1)):
                if num < probas_acum[i]:
                    muestra.append(soporte1[i])
                    break
    
                elif i == len(probas1)-1:
                    muestra.append(soporte1[i])
                    break
        return muestra
                
                
class Orden:
    def __init__(self, lista):
        self.lista = lista
    
    def ordenamiento_n(self, n):
        lista_a = self.lista.copy()
        ordenamiento = []
        for i in range(n):
            aleatorio = random.choice(lista_a)
            ordenamiento.append(aleatorio)
            lista_a.remove(aleatorio)
        return ordenamiento
    
    def combinacion(self):
        lista_a = self.lista.copy()
        n = len(lista_a)
        combinacion = []
        for i in range(n):
            aleatorio = random.choice(lista_a)
            combinacion.append(aleatorio)
            lista_a.remove(aleatorio)
        return combinacion
    
    def ordenamientos_portam(self, tam_grupos):
        lista_a = self.lista.copy()
        n = len(lista_a)
        ordenamientos = []
        
        # numero de grupos completos
        num_grupos = math.floor(n / tam_grupos)
        
        for i in range(num_grupos):
            grupo = Orden(lista_a).ordenamiento_n(tam_grupos)
            ordenamientos.append(grupo)
            
            for j in grupo:
                lista_a.remove(j)
            
        k = 0
        for restante in lista_a:
            if k == num_grupos:
                k = 0
            ordenamientos[k].append(restante)
            k += 1          
        return ordenamientos
    
    
class GSM4:
    def __init__(self,seed):
        self.semilla=seed
    def random(self):
        if self.semilla>=2**4:
            print(f"inserte una semilla menor a {2**4}")
        else:
            a=[]
            for i in range(0,4):
                r=self.semilla%2
                q=self.semilla//2
                a.append(r)
                self.semilla=q
            a.reverse()
            s1=a[0]+a[3]
            b=s1%2
            del a[0]
            a.append(b)
            x=0
            for i in range(0,len(a)):
                x+=a[i]*2**(len(a)-(i+1))
            self.semilla=x
            return x / 2**4
    
    
    def randoms(self,n):
        if n>=2**4:
            print("no es posible generar una muestra de tamaño: ",n)
        else:
            l=[]
            for i in range(0,n):
                l.append(self.random())
            return l

        
class GSM8:
    def __init__(self, seed):
        self.semilla = seed
    
    def random(self):
        if self.semilla >= 2 ** 8:
            print("inserte una semilla menor a ", 2 ** 8)
        else:
            a = []
            for i in range(0, 8):
                r = self.semilla % 2
                q = self.semilla // 2
                a.append(r)
                self.semilla = q
            a.reverse()
            s1 = a[0] + a[4] + a[5] + a[6]
            b = s1 % 2
            del a[0]
            a.append(b)
            x = 0
            for i in range(0, len(a)):
                x += a[i] * 2 ** (len(a) - (i + 1))
            self.semilla = x 
        return x / 2**8
    
    def randoms(self, n):
        if n >= 2 ** 8:
            print("no es posible generar una muestra de tamaño: ", n)
        else:
            l = []
            for i in range(0, n):
                l.append(self.random())
            return l

class GSM16:
    def __init__(self, seed):
        self.semilla = seed
    
    def random(self):
        if self.semilla >= 2 ** 16:
            print("inserte una semilla menor a ", 2 ** 16)
        else:
            a = []
            for i in range(0, 16):
                r = self.semilla % 2
                q = self.semilla // 2
                a.append(r)
                self.semilla = q
            a.reverse()
            s1 = a[0] + a[11] + a[13] + a[14]
            b = s1 % 2
            del a[0]
            a.append(b)
            x = 0
            for i in range(0, len(a)):
                x += a[i] * 2 ** (len(a) - (i + 1))
            self.semilla = x
        return x / 2**16
    
    def randoms(self, n):
        if n >= 2 ** 16:
            print("no es posible generar una muestra de tamaño: ", n)
        else:
            l = []
            for i in range(0, n):
                l.append(self.random())
            return l

class UnifCont:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def random(self):
        u = random.random()
        if self.a < self.b:
            x = (self.b - self.a) * u + self.a
        else:
            x = "a tiene que ser menor a b"
        return x

    def muestra(self, n):
        v = []
        for i in range(0, n):
            v.append(self.random())
        return v
    
    
class Exp:
    def __init__(self, lambd):
        self.l = lambd
    
    def random(self):
        u = random.random()
        x = -math.log(u) / self.l
        return x
    
    def muestra(self, n):
        v = []
        for i in range(0, n):
            v.append(self.random())
        return v

    
class Pois:
    def __init__(self, lambd):
        self.l = lambd
    
    def random(self):
        t = -math.log(random.random()) / self.l
        cont = 0
        while t < 1:
            t += -math.log(random.random()) / self.l
            cont += 1
        return cont
    
    def muestra(self, n):
        v = []
        for i in range(0, n):
            v.append(self.random())
        return v

class Norm:
    def __init__(self, miu, sigma):
        self.a = miu
        self.b = sigma
    
    def random(self):
        u1 = random.random()
        u2 = random.random()
        v1 = 2 * u1 - 1
        v2 = 2 * u2 - 1
        B = v1 ** 2 + v2 ** 2
        while B > 1:
            u1 = random.random()
            u2 = random.random()
            v1 = 2 * u1 - 1
            v2 = 2 * u2 - 1
            B = v1 ** 2 + v2 ** 2
        z1 = math.sqrt(-2 * math.log(B) / B) * v1 * self.b + self.a
        z2 = math.sqrt(-2 * math.log(B) / B) * v2 * self.b + self.a
        return [z1, z2]
    
    def muestra(self, n):
        v = []
        k = n % 2
        if k == 1:
            for i in range(0, n // 2):
                v.extend(self.random())
            v.append(self.random()[0])
        else:
            for i in range(0, n // 2):
                v.extend(self.random())
        return v

class Chisq:
    def __init__(self, df):
        self.df = df
    
    def Z2(self):
        u1 = random.random()
        u2 = random.random()
        v1 = 2 * u1 - 1
        v2 = 2 * u2 - 1
        B = v1 ** 2 + v2 ** 2
        while B > 1:
            u1 = random.random()
            u2 = random.random()
            v1 = 2 * u1 - 1
            v2 = 2 * u2 - 1
            B = v1 ** 2 + v2 ** 2
        z1 = math.sqrt(-2 * math.log(B) / B) * v1
        z2 = math.sqrt(-2 * math.log(B) / B) * v2
        return [z1 ** 2, z2 ** 2]
    
    def random(self):
        k = self.df % 2
        v = []
        if k == 1:
            for i in range(0, self.df // 2):
                v.extend(Chisq.Z2(self))
            v.append(Chisq.Z2(self)[0])
        else:
            for i in range(0, self.df // 2):
                v.extend(Chisq.Z2(self))
        return np.sum(v)
    
    def muestra(self, m):
        M = []
        for i in range(0, m):
            M.append(Chisq.random(self))
        return M


class Tstudent:
    def __init__(self, df):
        self.df = df
    
    def Z2(self):
        u1 = random.random()
        u2 = random.random()
        v1 = 2 * u1 - 1
        v2 = 2 * u2 - 1
        B = v1 ** 2 + v2 ** 2
        while B > 1:
            u1 = random.random()
            u2 = random.random()
            v1 = 2 * u1 - 1
            v2 = 2 * u2 - 1
            B = v1 ** 2 + v2 ** 2
        z1 = math.sqrt(-2 * math.log(B) / B) * v1
        z2 = math.sqrt(-2 * math.log(B) / B) * v2
        return [z1 ** 2, z2 ** 2]
    
    def random(self):
        k = self.df % 2
        v = []
        if k == 1:
            for i in range(0, self.df // 2):
                v.extend(Tstudent.Z2(self))
            v.append(Tstudent.Z2(self)[0])
        else:
            for i in range(0, self.df // 2):
                v.extend(Tstudent.Z2(self))
        u1 = random.random()
        u2 = random.random()
        v1 = 2 * u1 - 1
        v2 = 2 * u2 - 1
        B = v1 ** 2 + v2 ** 2
        while B > 1:
            u1 = random.random()
            u2 = random.random()
            v1 = 2 * u1 - 1
            v2 = 2 * u2 - 1
            B = v1 ** 2 + v2 ** 2
        z1 = math.sqrt(-2 * math.log(B) / B) * v1
        t = z1 / math.sqrt(np.sum(v) / self.df)
        return t
    
    def muestra(self, m):
        M = []
        for i in range(0, m):
            M.append(Tstudent.random(self))
        return M


class Fisher:
    def __init__(self, n, m):
        self.n = n
        self.m = m
    
    def Z2(self):
        u1 = random.random()
        u2 = random.random()
        v1 = 2 * u1 - 1
        v2 = 2 * u2 - 1
        B = v1 ** 2 + v2 ** 2
        while B > 1:
            u1 = random.random()
            u2 = random.random()
            v1 = 2 * u1 - 1
            v2 = 2 * u2 - 1
            B = v1 ** 2 + v2 ** 2
        z1 = math.sqrt(-2 * math.log(B) / B) * v1
        z2 = math.sqrt(-2 * math.log(B) / B) * v2
        return [z1 ** 2, z2 ** 2]
    
    def random(self):
        k = self.n % 2
        j = self.m % 2
        v = []
        w = []
        if k == 1:
            for i in range(0, self.n // 2):
                v.extend(Fisher.Z2(self))
            v.append(Fisher.Z2(self)[0])
        else:
            for i in range(0, self.n // 2):
                v.extend(Fisher.Z2(self))
        Xn = np.sum(v)
        if j == 1:
            for i in range(0, self.m // 2):
                w.extend(Fisher.Z2(self))
            w.append(Fisher.Z2(self)[0])
        else:
            for i in range(0, self.m // 2):
                w.extend(Fisher.Z2(self))
        Xm = np.sum(w)
        return ((Xn / self.n) / (Xm / self.m))
    
    def muestra(self, m):
        M = []
        for i in range(0, m):
            M.append(Fisher.random(self))
        return M

class Normacot:
    def __init__(self, acot):
        self.acot = acot
    
    def Z(self):
        u1 = random.random()
        u2 = random.random()
        v1 = 2 * u1 - 1
        v2 = 2 * u2 - 1
        B = v1 ** 2 + v2 ** 2
        while B > 1:
            u1 = random.random()
            u2 = random.random()
            v1 = 2 * u1 - 1
            v2 = 2 * u2 - 1
            B = v1 ** 2 + v2 ** 2
        z1 = math.sqrt(-2 * math.log(B) / B) * v1
        z2 = math.sqrt(-2 * math.log(B) / B) * v2
        return z1
    
    def random(self):
        z = Normacot.Z(self)
        while abs(z) > self.acot:
            z = Normacot.Z(self)
        return z
    
    def muestra(self, n):
        v = []
        for i in range(0, n):
            v.append(Normacot.random(self))
        return v



class lognormal:
    def __init__(self, miu, sigma):
        self.a = miu
        self.b = sigma
    
    def random(self):
        u1 = random.random()
        u2 = random.random()
        v1 = 2 * u1 - 1
        v2 = 2 * u2 - 1
        B = v1 ** 2 + v2 ** 2
        while B > 1:
            u1 = random.random()
            u2 = random.random()
            v1 = 2 * u1 - 1
            v2 = 2 * u2 - 1
            B = v1 ** 2 + v2 ** 2
        z1 = math.sqrt(-2 * math.log(B) / B) * v1 * self.b + self.a
        z2 = math.sqrt(-2 * math.log(B) / B) * v2 * self.b + self.a
        x = math.exp(z1)
        return x
    
    def muestra(self, n):
        v = []
        for i in range(0, n):
            v.append(lognormal.random(self))
        return v



class Gamma:
    def __init__(self, alpha):
        self.alpha = alpha
    
    def random(self):
        k = 0
        while random.random() > k:
            if self.alpha > 0 and 1 > self.alpha:
                p = random.random()
                if p <= math.e / (math.e + self.alpha):
                    y = ((math.e + self.alpha) * p / math.e) ** (1 / self.alpha)
                else:
                    y = -math.log((math.e + self.alpha) * (1 - p) / math.e * self.alpha)
                if y <= 0:
                    fy = 0
                    fx = 0
                elif y > 0 and y <= 1:
                    fy = math.e * self.alpha * y ** (self.alpha - 1) / (math.e + self.alpha)
                    fx = (y ** (self.alpha - 1)) * (math.e ** (-y)) / math.gamma(self.alpha)
                else:
                    fy = math.e * self.alpha * math.exp(-y) / (math.e + self.alpha)
                    fx = (y ** (self.alpha - 1)) * (math.e ** (-y)) / math.gamma(self.alpha)
                t = (self.alpha + math.e) / math.e * self.alpha * math.gamma(self.alpha)
            else:
                p = random.random()
                a = math.sqrt(2 * self.alpha - 1)
                b = 4 * math.exp(-self.alpha) * self.alpha ** self.alpha / (a * math.gamma(self.alpha))
                y = (p * self.alpha ** a / (1 - p)) ** (1 / a)
                if y <= 0:
                    fy = 0
                    fx = 0
                    t = 0
                else:
                    fy = ((y ** a) / (y ** a + self.alpha ** a)) * a * self.alpha ** a
                    fx = (y ** (self.alpha - 1)) * (math.e ** (-y)) / math.gamma(self.alpha)
                    t = b * (y ** (a - 1)) / ((self.alpha ** a) + (y ** a)) ** 2
            k = fx / t * fy
        return y

    def muestra(self, n):
        v = []
        for i in range(0, n):
            v.append(Gamma.random(self))
        return v



class SDE:
    def __init__(self, parametros, s0, mu, sigma):
        self.parametros = parametros
        self.s0 = s0
        self.mu = mu
        self.sigma = sigma
    
    def euler(self, T, n, n_simulaciones):
        plt.figure(figsize=(15, 5))
        dt = T / n
        
        trayectorias = []
        for j in range(n_simulaciones):
            valores = Norm(0, 1).muestra(n)
            proceso = [self.s0]
            valores_dt = [dt]
            s0_sim = self.s0
            delta_t = dt

            for i in range(1, len(valores)):
                si = s0_sim + mu.subs({**parametros, S:s0_sim, t:delta_t})*dt + sigma.subs({**parametros, S:s0_sim, t:delta_t})*valores[i]*np.sqrt(dt)
                proceso.append(si)
                s0_sim = si
                delta_t += dt
                valores_dt.append(dt*(i+1))
            x = np.arange(1, len(proceso)+1)
            
            trayectorias.append(proceso)
            plt.plot(valores_dt, proceso)
        
        plt.title('Euler simulation', fontsize=16)
        plt.xlabel('Tiempos', fontsize=12)
        plt.ylabel('Precio', fontsize=12)
        plt.grid()
        return trayectorias

        
    def milstein(self, T, n, n_simulaciones):
        plt.figure(figsize=(15, 5))
        dt = T / n
        trayectorias = []
        for j in range(n_simulaciones):
            valores = Norm(0, 1).muestra(n)
            proceso = [self.s0]
            valores_dt = [dt]
            s0_sim = self.s0
            delta_t = dt
            for i in range(1, len(valores)):
                si = s0_sim + mu.subs({**parametros, S:s0_sim,  t:delta_t})*dt + sigma.subs({**parametros, S:s0_sim,  t:delta_t})*valores[i]*np.sqrt(dt) + 1/2 * sp.diff(sigma, S).subs({**parametros, S:s0_sim, t:delta_t}) * sigma.subs({**parametros, S:s0_sim, t:delta_t}) * dt * (valores[i]**2 - 1)          
                proceso.append(si)
                s0_sim = si
                delta_t += dt
                valores_dt.append(dt*(i+1))
            x = np.arange(1, len(proceso)+1)
            trayectorias.append(proceso)
            plt.plot(valores_dt, proceso)
            
        plt.title('Milstein simulation', fontsize=16)
        plt.xlabel('Tiempos', fontsize=12)
        plt.ylabel('Precio', fontsize=12)
        plt.grid()
        return trayectorias


class Poisson_homogeneo:
    def __init__(self, T, dt, lam):
        self.T = T
        self.dt = dt
        self.lam = lam

    def trayectoria(self):
        m = math.ceil(self.T / self.dt)
        mt = np.array(0).reshape(1, 1)
        tiempos = []
        while True:
            tiempos.append(Exp(self.lam).random())
            if sum(tiempos) > self.T:
                tiempos.pop(-1)
                break
        tiempos_buenos = []
        for g in range(len(tiempos)):
            tiempos_buenos.append(sum(tiempos[0:g]))
        tiempos_buenos.append(self.T)

        saltos = np.concatenate((np.array([0]), np.arange(0, len(tiempos))))
        return tiempos_buenos, saltos

    def simulaciones(self, n):
        plt.figure(figsize=(10, 5))
        for i in range(n):
            x, y = Poisson_homogeneo.trayectoria(self)
            plt.step(x, y)
        plt.grid()


class Poisson_no_homogeneo:
    def __init__(self, T, max):
        self.T = T
        self.max = max

    def trayectoria(self):
        mt = np.array(0).reshape(1, 1)
        times = np.array(0).reshape(1, 1)
        while float(np.sum(mt)) < self.T:
            u = uniform(0, 1).rvs()
            v = uniform(0, 1).rvs()
            t = -math.log(u) / self.max
            if v < t:
                mt = np.concatenate((mt, np.array(t).reshape(1, 1)), axis=0)
                times = np.concatenate((times, np.array(float(np.sum(mt))).reshape(1, 1)), axis=0)
        tiempo_saltos = np.concatenate((np.concatenate((times, np.cumsum(mt).reshape(len(mt), 1)), axis=1)[0:len(mt) - 1, :][:, 0], np.array([self.T])))
        valores_saltos = np.arange(0, len(tiempo_saltos))
        return tiempo_saltos, valores_saltos
    
    
    def simulaciones(self, n):
        plt.figure(figsize=(10, 5))
        for i in range(n):
            x, y = Poisson_no_homogeneo.trayectoria(self)
            plt.step(x, y)
        plt.grid()


    
class Poisson_compuesto:
    def __init__(self, T, dt, l):
        self.T = T
        self.dt = dt
        self.l = l

    def trayectoria(self):
        tiempos = []
        while True:
            tiempos.append(Exp(self.l).random())
            if sum(tiempos) > self.T:
                tiempos.pop(-1)
                break
        tiempos_buenos = []
        for g in range(len(tiempos)):
            tiempos_buenos.append(sum(tiempos[0:g]))
        tiempos_buenos.append(self.T)
        
        m = math.ceil(self.T / self.dt)
        mt = np.array(0).reshape(1, 1)
        tray = np.cumsum(np.array(poisson(self.l * self.dt).rvs(len(tiempos_buenos)-2))).reshape(len(tiempos_buenos)-2, 1)
        saltoss = np.concatenate((mt, tray), axis=0).ravel()
        saltos = np.concatenate((np.array([0]),saltoss))
        return tiempos_buenos, saltos 
    

    def simulaciones(self, n):
        plt.figure(figsize=(10, 5))
        for i in range(n):
            x, y = Poisson_compuesto.trayectoria(self)
            plt.step(x, y)
        plt.grid()



def wiener(simulaciones, n, t):
    trayectorias = []
    for i in range(simulaciones):
        dt = 1 / n
        dx = 1 / np.sqrt(n)
        p = 0.5
        q = 1 - p
        tiempos = np.arange(0, math.floor(n*t)+1)*dt
        X = random.choices([-1, 1], [q, p], k=math.floor(n*t))
        W_0 = 0
        W_t = [W_0]
        W_t.extend(np.cumsum(X)*dx)
        trayectorias.append(np.array(W_t))

    trayectorias = np.array(trayectorias)


    plt.figure(figsize=(15,5))
    for i in range(simulaciones):
        plt.plot(tiempos, trayectorias[i], lw=1, ms=1)
    plt.grid()


class Bernoulli:
    def __init__ (self,p):
        self.p=p
        
    
    def numero (self):
        u=random.random()
        if u<self.p:
            return 1
        else:
            return 0
    
    def lista (self,n):
        List=[]
        for i in range (1,n+1):
            List.append(self.numero())
        return List
class ValOpc:
    def __init__(self, r, sigma, dt, T, S0):
        self.r = r
        self.sigma = sigma
        self.dt = dt
        self.T = T
        self.S0 = S0

    def rand(self):
        N = int(self.T/self.dt)
        u = np.exp(self.sigma*np.sqrt(self.dt))
        d = 1/u
        p = (np.exp(self.r*self.dt)-d)/(u-d)

        precio = np.zeros(N+1)
        precio[0] = self.S0
        generador = Bernoulli(0.5)
        for i in range(1, N+1):
            bernoulli = generador.numero()
            if bernoulli == 1:
                precio[i] = u*precio[i-1]
            else:
                precio[i] = d*precio[i-1]

        time = np.cumsum(np.concatenate(([0], [self.dt for i in range(N)])))
        plt.plot(time, precio, color='red')
        plt.grid()
        return precio[len(precio)-1]

    def muestra(self, n):
        X = []
        for i in range(n):
            Y = ValOpc(self.r, self.sigma, self.dt, self.T, self.S0)
            X.append(Y.rand())
        plt.grid()
        return np.mean(X)
    







class series:
    def __init__(self, confianza, muestra, d, k):
        self.d = d
        self.confianza = confianza
        self.k = k
        self.muestra = np.array(muestra)
        self.n = len(self.muestra)
    def prueba(self):
        m = self.n // self.d
 
        # Verificamos que el tamaño de m sea el adecuado
        if (self.k ** self.d) * 10 <= m and m <= (self.k ** self.d) * 100:
            r = self.n % self.d
 
            # Quitamos valores de la muestra para que se pueda cambiar su forma
            if r != 0:
                if r == 1:
                    muest = np.delete(self.muestra, self.n - 1)
                else:
                    muest = np.delete(self.muestra, (self.n - 1, self.n - 2))
                U = np.reshape(muest, (m, self.d))
            else:
                U = np.reshape(self.muestra, (m, self.d))
 
            # Averiguamos el tamaño de la dimensión
            # Dimensión 2
            if self.d == 2:
                I = np.zeros((self.k, self.k))
                for i in range(0, m):
                    I[math.floor(U[i, 0] * self.k), math.floor(self.k * U[i, 1])] += 1
                F = np.reshape(I, (1, self.k ** self.d))
                t = np.sum((F - (m / self.k ** self.d)) ** 2) * (self.k ** self.d / m)
                alpha = 1 - self.confianza
                valor_c = stats.chi2.ppf(q=1 - alpha, df=self.k ** self.d - 1)
                if t > valor_c:
                    resp = 'se rechaza la hipótesis'
                else:
                    resp = 'no se rechaza la hipótesis'
            # Dimensión 3
            else:
                I = np.zeros((self.k, self.k, self.k))
                for i in range(0, m):
                    I[math.floor(U[i, 0] * self.k), math.floor(self.k * U[i, 1]), math.floor(self.k * U[i, 2])] += 1
                F = np.reshape(I, (1, self.k ** self.d))
                t = np.sum((F - (m / self.k ** self.d)) ** 2) * (self.k ** self.d / m)
                alpha = 1 - self.confianza
                valor_c = stats.chi2.ppf(q=1 - alpha, df=self.k ** self.d - 1)
                if t > valor_c:
                    resp = 'se rechaza la hipótesis'
                else:
                    resp = 'no se rechaza la hipótesis'
 
        # Caso distinto
        else:
            resp = "eliga un tamaño de muestra adecuado"
            t = "eliga un tamaño de muestra adecuado"
            valor_c = "eliga un tamaño de muestra adecuado"
 
        return resp, t, valor_c

def prueba_correlacion(U,desplazamiento,confianza):
    n = len(U)
    h = math.floor((n-1-desplazamiento)/desplazamiento)

    #Estimadores
    suma = 0
    for k in range(h+1):
        suma += U[k*desplazamiento]*U[(k+1)*desplazamiento]
    estimador_cov = 1/(h+1)*suma-1/4
    estimador_corr = 12*estimador_cov
    var_estimador_corr = (13*h+7)/((h+1)**2)

  #Estadístico
    alfa = 1-confianza
    estadistico = estimador_corr/np.sqrt(var_estimador_corr)
    Z = stats.norm.ppf(1-alfa/2)

    #Prueba de Hipótesis
    if abs(estadistico) < Z:
        return "No se rechaza H0"
    else:
        return "Se rechaza H0"


class frecuencia:
    def __init__(self,confianza,muestra,k):
        self.conf=confianza
        self.m=np.array(muestra)
        self.n=len(self.m)
        self.k=k
    def prueba(self):
        if self.k*10<= self.n and self.n<=self.k*100:
            alpha=1-self.conf
            increm=1/self.k
            f=np.zeros(self.k)
            acum=[]
            for i in range(0,self.n):
                boleano=1
                interv=0
                cont=0
                while boleano==1:
                    if interv<=self.m[i]<=interv+increm:
                        boleano=0
                    else:
                        boleano=1
                    interv+=increm
                cont+=1
                acum.append(cont)
            for i in range(0,self.k):
                f[i]=acum.count(i+1)
            t=np.sum((f-(self.n/self.k))**2)*(self.k/self.n)
            valor_c=stats.chi2.ppf(q=1- alpha, df=self.k-1)
            if t>valor_c:
                resp='se rechaza la hipótesis'
            else:
                resp='no se rechaza la hipótesis'
        else:
            resp="eliga un tamaño de muestra adecuado"
            t="eliga un tamaño de muestra adecuado"
            valor_c="eliga un tamaño de muestra adecuado"
        return (resp,t,valor_c)



class B_S:
    def __init__(self, S_0, K, t_0, T, r, sigma_estimation, data):
        self.S_0 = S_0
        self.K = K
        self.t_0 = datetime.strptime(t_0, '%Y-%m-%d')
        self.T = datetime.strptime(T, '%Y-%m-%d')
        self.r = r
        self.sigma_estimation = sigma_estimation
        self.data = data
        self.rowss = data.shape[0]

    
    def sigma(self):
        if self.sigma_estimation == 'hist':
            return np.sqrt(((self.rowss - 1) / self.rowss) * np.var(self.data['Log Returns'].values, ddof=1))
        elif self.sigma_estimation == 'density':
            n_returns = self.data.shape[0] - 1
            s2 = np.var(self.data[stock_data.columns[-1]].values, ddof=1)
            s = np.std(self.data[stock_data.columns[-1]].values, ddof=1)
            E_s2 = ((n_returns - 1)*s2) / (n_returns - 3)
            return np.sqrt(E_s2)
        else:
            raise ValueError('Método de estimación de sigma no válido')

    
    def call(self, how='single'):
        if how == 'single':
            time_until_maturity = (self.T - self.t_0).days / 365.25
        elif how == 'nel':
            days = (self.T-self.t_0).days
            time_until_maturity = np.linspace(days, 0.000001, 100) / 365.25
        elif how == 'precios':
            precio_min = self.S_0 - 20
            precio_max = self.S_0 + 20
            precios = np.linspace(precio_min, precio_max, 100)
            time_until_maturity = (self.T - self.t_0).days / 365.25
            d1 = (np.log(precios/self.K) + (self.r + B_S.sigma(self)**2 /2)*(time_until_maturity)) / (B_S.sigma(self)*np.sqrt(time_until_maturity))
            d2 = d1 - B_S.sigma(self)*np.sqrt(time_until_maturity)
            return precios * norm.cdf(d1) - self.K * np.exp(-self.r * time_until_maturity) * norm.cdf(d2)
        d1 = (np.log(self.S_0/self.K) + (self.r + B_S.sigma(self)**2 /2)*(time_until_maturity)) / (B_S.sigma(self)*np.sqrt(time_until_maturity))
        d2 = d1 - B_S.sigma(self)*np.sqrt(time_until_maturity)
        return self.S_0 * norm.cdf(d1) - self.K * np.exp(-self.r * time_until_maturity) * norm.cdf(d2)

    
    def put(self, how='single'):
        if how == 'single':
            time_until_maturity = (self.T - self.t_0).days / 365.25
        elif how == 'nel':
            days = (self.T-self.t_0).days
            time_until_maturity = np.linspace(days, 0.000001, 100) / 365.25

        d1 = np.log((self.S_0/self.K) + (self.r + B_S.sigma(self)/2)*(time_until_maturity)) / (B_S.sigma(self)*np.sqrt(time_until_maturity))
        d2 = d1 - B_S.sigma(self)*np.sqrt(time_until_maturity)
        return self.K * np.exp(-self.r * time_until_maturity) * norm.cdf(-d2) - (self.S_0 * norm.cdf(-d1))
    

    def plot_time(self):
        prices_call = B_S.call(self, 'nel')
        prices_put = B_S.put(self, 'nel')
        days = (self.T-self.t_0).days / 365.25
        time_until_maturity = np.linspace(days, 0.0, 100)
        plt.figure(figsize=(15, 9))
        plt.plot(time_until_maturity, prices_call, label='Call Option')
        plt.plot(time_until_maturity, prices_put, label='Put Option')
        plt.xlabel('Time to Maturity (Years)')
        plt.ylabel('Option Price')
        plt.title('Black-Scholes Option Pricing Model')
        plt.legend()
        plt.grid(True)
        plt.show()

    
    def plot_prices(self):
        precios_call = []
        precios_put = []
        precio_min = self.S_0 - self.S_0/2
        precio_max = self.S_0 + self.S_0/2
        precios = np.linspace(precio_min, precio_max, 100) 
        for i in precios:
            precios_call.append(B_S(i, self.K, self.t_0.strftime('%Y-%m-%d'), self.T.strftime('%Y-%m-%d'), self.r, self.sigma_estimation, self.data).call())
            precios_put.append(B_S(i, self.K, self.t_0.strftime('%Y-%m-%d'), self.T.strftime('%Y-%m-%d'), self.r, self.sigma_estimation, self.data).put())
        
        plt.figure(figsize=(15, 9))
        plt.plot(precios, precios_call, label='Call Option')
        plt.plot(precios, precios_put, label='Put Option')
        plt.axvline(self.K, color='red', linestyle='--', label='Strike Price')
        plt.xlabel('Stock price')
        plt.ylabel('Option Price')
        plt.title('Black-Scholes')
        plt.legend()
        plt.grid(True)
        plt.show()

    
    def plot_profit(self):
        premium = 0.1 * self.S_0
        precio_min = self.S_0 - 10
        precio_max = self.S_0 + 30
        precios = np.linspace(precio_min, precio_max, 100) 
        plt.figure(figsize=(15, 9))
        plt.plot(precios, 0*precios, color='black')
        plt.axvline(self.K, color='red', linestyle='--', label='Strike Price')
        plt.plot(precios, np.maximum(0, precios - self.K ), label='Call payoff', color='lightblue', linestyle='--')
        plt.plot(precios, premium - np.maximum(0, precios - self.K), label='Put payoff', color='green')
        plt.plot(precios, np.maximum(0, precios - self.K ) - premium, label='Call profit', color='blue')
        plt.plot(precios, -np.maximum(0, precios - self.K), label='Put profit', linestyle='--', color='lightgreen')
        plt.yticks([premium, -premium], labels=['premium', '- premium'])
        plt.legend(loc='best', fontsize=15, ncol=1)
        plt.grid()



class SAF:
    def __init__(self,vects,probs):
        self.vects = vects
        self.probs = probs
        self.n = len(self.vects)

    def rand(self):
        u = random.random()
        s = 0
        for i in range (0,self.n):
            if s <= u <= self.probs[i] + s:
                return self.vects[i]
            s+=self.probs[i]
        
    def muestra(self,n):
        v=[]
        for i in range(0,n):
            v.append(self.rand())
        return v

class Linesp:
    def __init__(self,lamda,miu,horizonte,servidores):
        self.m=miu
        self.lam=lamda
        self.h=horizonte
        self.s=servidores
    
    def tray(self):
        i=0
        t=0
        T=[0]
        U=[0]
        C=[0]
        while t<self.h:
            T_i=Exp(self.m+self.lam).muestra(1)
            t+=T_i[0]
            T.append(T_i[0])
            if U[i]>0:
                if U[i]>self.s & U[i]<=2*self.s:
                    paso=SAF([1,-1],[self.lam/(self.lam+(U[i]-self.s)*self.m),self.m*(U[i]-self.s)/(self.lam+(U[i]-self.s)*self.m)]).muestra(1)[0] 
                    
                if U[i]>2*self.s:
                    paso=SAF([1,-1],[self.lam/(self.lam+(self.s)*self.m),self.m*(self.s)/(self.lam+(self.s)*self.m)]).muestra(1)[0]
                    
                if U[i]<=self.s:
                    paso=SAF([1,-1],[self.lam/(self.lam+self.m),self.m/(self.lam+self.m)]).muestra(1)[0]
            else:
                paso=1
              
            C.append(paso)
            U.append(U[i]+paso)
            i+=1
        return T,U,C

    def stats(self,tipo):
        T,U,C=self.tray()
        ac_1=0
        con_1=0
        p_1=0
        
        ac_2=0
        con_2=0
        p_2=0
        
        if tipo=="FIFO":
            for i in range (0,len(T)):
                if U[i]>self.s:
                    ac_1+=T[i]
                    con_1+=1
                    
            if con_1!=0:
                p_1=ac_1/con_1
            else:
                p_1=0
            
            for i in range (0,len(T)):
                if C[i]==-1:
                    ac_2+=T[i]
                    con_2+=1
                    
            if con_2!=0:
                p_2=ac_2/con_2
            else:
                p_2=0
        
        if tipo=="LIFO":
            L_1=[0]
            
            for i in range (0,len(T)):
                if U[i]>self.s:
                    L_1.append(T[i])
                    con_1+=1
                    
            L_2=np.cumsum(L_1)
            
            for i in range(0,len(L_2)):
                ac_1+=L_2[i]
            p1=ac_1/con_1
            if con_1!=0:
                p_1=ac_1/con_1
            else:
                p_1=0
            
            for i in range (0,len(T)):
                if C[i]==-1:
                    ac_2+=T[i]
                    con_2+=1
                    
            if con_2!=0:
                p_2=ac_2/con_2
            else:
                p_2=0
        
        return ("El tiempo promedio de espera es: {} ".format(p_1),
                "El tiempo promedio en el servidor es: {}".format(p_2))



class Var_estocastico:
    def __init__(self, returns):
        self.returns = returns

    def VaR(self, confianza=0.95, n=10000):
        mu = np.mean(returns)
        sigma2 = ((len(returns) - 1) / len(returns)) * np.var(returns)
        simulaciones = Norm(mu, sigma2).muestra(n) 
        VaR = np.percentile(simulaciones, 1-confianza)
        return VaR
