import numpy as np
import math
import random
import matplotlib.pyplot as plt
import time
from scipy import stats

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
#         self.x00_s = np.array(x00_s)
#         self.y00_s = np.array(y00_s)
        
        
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
    
    def randoms(self):
        nums = []
        xi_anterior = self.x00
        yi_anterior = self.y00
        for i in range(self.m1 - 1):
            xi = np.dot(xi_anterior, self.b1_s) % self.m1
            yi = (yi_anterior*self.b2) % self.m2
            zi = (xi - yi) % self.m1
            
            if zi/self.m1 in nums: break
            else: 
                nums.append(zi/self.m1)
                xi_anterior = xi   # actualizando la x(i-1)
                yi_anterior = yi   # actualizando la y(i-1)
                
        return nums
    
    
    
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
        num = random.random()

        muestra = []
        for _ in range(N):
            for i in range(len(probas1)):
                if num < probas_acum[i]:
                    muestra.append(soporte1[i])
                    
                elif i == len(probas1)-1:
                    muestra.append(soporte1[i])
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
        num = random.random()

        
        muestra = []
        for _ in range(n):
            for i in range(len(probas1)):
                if num < probas_acum[i]:
                    muestra.append(soporte1[i])
                    
                elif i == len(probas1)-1:
                    muestra.append(soporte1[i])
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
        num = random.random()
        muestra = []
        for _ in range(n):
            for i in range(len(probas1)):
                if num < probas_acum[i]:
                    muestra.append(soporte1[i])
                    
                elif i == len(probas1)-1:
                    muestra.append(soporte1[i])
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
        num = random.random()

        muestra = []
        for _ in range(m):
            for i in range(len(probas1)):
                if num < probas_acum[i]:
                    muestra.append(soporte1[i])
                    
                elif i == len(probas1)-1:
                    muestra.append(soporte1[i])
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
        z = Normacot.Z()
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
            valores = normal(0, 1).randoms(n)
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
            valores = normal(0, 1).randoms(n)
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


class poiss_process:
    def __init__(self,l,t,sim_ts):
        self.l=l
        self.t=t
        self.sim_ts=sim_ts
        
    def rand(self):
        t1=-math.log(random.random())/self.l
        cont=0
        acm=t1
        lista=[]
        while acm<=self.t:
            lista.append(acm)
            t2=-math.log(random.random())/self.l
            acm+=t2
            cont+=1
        if self.sim_ts==True:
            return lista
        else:
            return cont
    
    def sample(self,n):
        self.sim_ts==False
        lista = []
        for i in range(n):
            lista.append(self.rand())

        return lista


class poiss_nohom:
    def __init__(self,T,max,sim_ts):
        self.T=T
        self.max=max
        self.sim_ts=sim_ts
        
    def rand(self):
        lista = []
        acm = 0
        cont = 0
        while acm < self.T:
            u=random.random()
            v=random.random()
            t=-math.log(u)/self.max
            if v<t :
                acm+=t
                lista.append(t)
                cont+=1
        if self.sim_ts==True:
            return lista
        else:
            return cont
        
    def sample(self,n):
        self.sim_ts==False
        lista=[]
        for i in range(n):
            lista.append(self.rand())
        return lista
    
class poiss_comp:
    def __init__(self,l,t,s):
        self.l=l
        self.t=t
        self.s=s
    def rand(self):
        n=poiss_process(self.t,self.l,sim_ts=False).rand()
        z=Norm(0,self.s).muestra(n)
        return np.sum(z)
    def sample(self,n):
        lista=[]
        for i in range(n):
            lista.append(self.rand())
        return lista

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




class Valuacion_Opciones:
    def __init__(self, r, sigma, dt, T, S0):
        self.r = r
        self.sigma = sigma
        self.dt = dt
        self.T = T
        self.S0 = S0

    def precio(self):
        N = int(self.T/self.dt)
        u = np.exp(self.sigma*np.sqrt(self.dt))
        d = 1/u
        p = (np.exp(self.r*self.dt)-d)/(u-d)

        precio = np.zeros(N+1)
        precio[0] = self.S0
        for i in range(1, N+1):
            bernoulli = Numgen.bernoulli(p, 1)
            if bernoulli == 1:
                precio[i] = u*precio[i-1]
            else:
                precio[i] = d*precio[i-1]

        time = np.cumsum(np.concatenate(([0], [self.dt for i in range(N)])))
        plt.grid()
        plt.plot(time, precio, color='r')
        return precio[len(precio)-1]

    def muestra(self, n):
        X = []
        for i in range(n):
            Y = Valuacion_Opciones(self.r, self.sigma, self.dt, self.T, self.S0)
            X.append(Y.precio())
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


