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

    def masapoisson():
        pass
    

    
class Numgen:
    def unif(a, b):
        lista = [x for x in range(a, b)]
        return random.choice(lista)

    def unif_arbitrario(lista):
        return random.choice(lista)

    def bernoulli(p):
        num = random.random()
        if num < p:
            return 1
        elif num > p:
            return 0

    def binomial(n, p):
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

        for i in range(len(probas1)):
            if num < probas_acum[i]:
                return soporte1[i]
                
            elif i == len(probas1)-1:
                return soporte1[i]
                

    def geom(p):
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

        for i in range(len(probas1)):
            if num < probas_acum[i]:
                return soporte1[i]
                
            elif i == len(probas1)-1:
                return soporte1[i]
                

    def nbinomial(r, p):
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

        for i in range(len(probas1)):
            if num < probas_acum[i]:
                return soporte1[i]
                
            elif i == len(probas1)-1:
                return soporte1[i]
                

    def hyper(N, k, n):    
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

        for i in range(len(probas1)):
            if num < probas_acum[i]:
                return soporte1[i]
                
            elif i == len(probas1)-1:
                return soporte1[i]
                
                
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
    
    
    def muestra(self,n):
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
    
    def muestra(self, n):
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
    
    def muestra(self, n):
        if n >= 2 ** 16:
            print("no es posible generar una muestra de tamaño: ", n)
        else:
            l = []
            for i in range(0, n):
                l.append(self.random())
            return l

class GSM32:
    def __init__(self, seed):
        self.semilla = seed
    
    def random(self):
        if self.semilla >= 2 ** 32:
            print("inserte una semilla menor a ", 2 ** 32)
        else:
            a = []
            for i in range(0, 32):
                r = self.semilla % 2
                q = self.semilla // 2
                a.append(r)
                self.semilla = q
            a.reverse()
            s1 = a[0] + a[25] + a[27] + a[29] + a[30] + a[31]
            b = s1 % 2
            del a[0]
            a.append(b)
            x = 0
            for i in range(0, len(a)):
                x += a[i] * 2 ** (len(a) - (i + 1))
            self.semilla = x
        return x / 2**32
    
    def muestra(self, n):
        if n >= 2 ** 32:
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

class Jicuad:
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
                v.extend(Jicuad.Z2(self))
            v.append(Jicuad.Z2(self)[0])
        else:
            for i in range(0, self.df // 2):
                v.extend(Jicuad.Z2(self))
        return np.sum(v)
    
    def muestra(self, m):
        M = []
        for i in range(0, m):
            M.append(Jicuad.random(self))
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
        z = Nacot.Z()
        while abs(z) > self.acot:
            z = Normacot.Z(self)
        return z
    
    def muestra(self, n):
        v = []
        for i in range(0, n):
            v.append(Nacot.random(self))
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
#     params: parametros iniciales en un diccionario
#     s0: valor inicial del proceso
#     mu, sigma: deriva y difución en sympy
    def __init__(self, parametros, s0, mu, sigma):
        self.parametros = parametros
        self.T = T
        self.n = n
        self.s0 = s0
        self.mu = mu
        self.sigma = sigma
    
    def euler(self, T, n, n_simulaciones):
        
        dt = T / n
        
        
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
            plt.plot(valores_dt, proceso)
        plt.grid
        
    def milstein(self, T, n, n_simulaciones):

        dt = T / n
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
            plt.plot(valores_dt, proceso)
        plt.grid()
