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


