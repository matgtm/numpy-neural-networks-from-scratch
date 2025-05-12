import numpy as np



class NN:
    
    def __init__(self, input_size,
                 output_size, 
                 neurons_per_layer,
                 lr0=0.01,
                 optimizer='SGD',
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-8,
                 batch_size=64,
                 l2=0.01,
                 early_stop_patience=10,
                 use_dropout=False,
                 p_dropout=0.5):
        
        # Dimensiones
        self.input_size = input_size
        self.output_size = output_size
        rows_per_matrix = neurons_per_layer
        self.mtx_dims = self.get_mtx_dims(rows_per_matrix, input_size, output_size)
        
        # Diccionarios: pesos, deltas y resultados (z y a), por capa 
        self.weights = {}
        self.grads = {}
        self.cache = {}
        self.deltas = {}
        self._init_weights()
        if optimizer.lower() == 'sgd':
            self.optimizer = 'sgd'
        elif optimizer.lower() == 'adam':
            self.init_adam(beta_1, beta_2, epsilon)
        
        # Hiperparámetros de entrenamiento
        self.lr0 = lr0
        self.batch_size = batch_size
        self.l2 = l2
        self.early_stop_patience = early_stop_patience
        self.use_dropout   = use_dropout
        if use_dropout:
            self.p_dropout = p_dropout
        elif not use_dropout:
            self.p_dropout = 1.0
           
    def get_mtx_dims(self, rows_per_matrix, input_size, output_size):
        cols_per_matrix = [input_size] + rows_per_matrix
        rows_per_matrix = rows_per_matrix + [output_size]
        return list(zip(rows_per_matrix, cols_per_matrix))
         
    def _init_weights(self):
        np.random.seed(50)
        n_cols = self.input_size
        for i, (n_rows, n_cols) in enumerate(self.mtx_dims): #Por cada capa
            # Inicializo W y b con 'He'
            W = np.random.randn(n_rows, n_cols) * np.sqrt(2 / n_cols)
            b = np.zeros((n_rows, 1))
            # Inicializo diccionarios de pesos y gradientes
            self.weights[f'W{i+1}'] = W
            self.weights[f'b{i+1}'] = b
            self.grads[f'dW{i+1}'] = np.zeros_like(W)
            self.grads[f'db{i+1}'] = np.zeros_like(b)
            
    """pseudo fit NN simple
    train_loss, val_loss = [], []
    por cada epoch:
        gradients = []
        por cada dato en X_train:
            y_hat = forward(X)
            loss = compute_loss(y_hat, y_true)
            gradient = backward(X, y_hat, y_true)
            gradients.append(gradient)
            train_loss[epoch] += loss
        train_loss[epoch] /= len(X_train)
        update_weights(gradients)
        
        por cada dato en X_val:
            y_hat = forward(X)
            loss = compute_loss(y_hat, y_true)
            val_loss[epoch] += loss
        val_loss[epoch] /= len(X_val)
    """
    def fit(self,
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=100,
            schedule_type=None, 
            lr_final=None, 
            decay_lambda=None):
        
        # Historiales de loss
        train_loss, val_loss = [], []
        # Reinicio adam
        if self.optimizer == 'adam':
            self.adam_reset()

        self._min_val_loss_for_best_weights = float('inf')
        self.best_weights = {k: v.copy() for k, v in self.weights.items()} # Initialize with current weights

        # recorro epochs    
        for epoch in range(epochs): 
            suma_loss_train = 0.0
            
            # Learning rate schedule
            if schedule_type == 'linear':
                self.lr = self.lr0 - (self.lr0 - lr_final) * (epoch / epochs)
            elif schedule_type == 'exponential':
                self.lr = self.lr0 / np.exp(decay_lambda * epoch)
            else:
                self.lr = self.lr0
            
            # Barajo los datos
            N = X_train.shape[0]
            idx = np.random.permutation(N)
            X_train_shuffled = X_train[idx]
            y_train_shuffled = y_train[idx]
            for start in range(0, N, self.batch_size): # recorro batches
                end = min(start + self.batch_size, N)
                X_batch = X_train_shuffled[start:end]
                y_batch = y_train_shuffled[start:end]
                
                for i, x in enumerate(X_batch): # recorro datos dentro del batch
                    y_pred = self.forward(x, training=True)    # vector probabilidades
                    self.backward(x, y_pred, y_batch[i])   # vector gradiente
                    # loss
                    loss = self.loss(y_pred, y_batch[i])  # escalar
                    suma_loss_train += loss
                # Actualizar pesos
                self.update_weights()
                self.reset_gradients() # reinicio gradientes
                
            
            train_loss.append(suma_loss_train/len(X_train))
            print(f"Epoch {epoch+1}/{epochs}, Train loss: {train_loss[epoch]}")
            
            suma_loss_val = 0.0
            for i, x in enumerate(X_val): # recorro datos val
                y_pred = self.forward(x)    # vector probabilidades
                loss = self.loss(y_pred, y_val[i])  # escalar
                suma_loss_val += loss
            val_loss.append(suma_loss_val/len(X_val))
            
            # Guardo mejores pesos
            if val_loss[epoch] < self._min_val_loss_for_best_weights:
                self._min_val_loss_for_best_weights = val_loss[epoch]
                self.best_weights = {k: v.copy() for k, v in self.weights.items()}
                
            # Early stopping
            if epoch > 0 and val_loss[epoch] > val_loss[epoch-1]:
                if epoch - np.argmin(val_loss) >= self.early_stop_patience:
                    # Restore best weights
                    self.weights = self.best_weights
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        history = {
            'train_loss': train_loss,
            'val_loss': val_loss
        }
        return history
    
    def predict(self, X):
        N = X.shape[0]
        y_pred = np.zeros(N, dtype=int)
        for i, x in enumerate(X):
            probs = self.forward(x)    #probas     
            y_pred[i] = np.argmax(probs)    #prediccion
        return y_pred
    
    def forward(self, x, training=False):
        self.cache.clear()
        L = len(self.mtx_dims) # L = ultima capa = cant capas
        a_prev = x.reshape(-1, 1) 
        self.cache[f'a{0}'] = a_prev
        
        for l in range(1,L+1): # l = 1..L 

            # Cálculos matriciales por capa
            W = self.weights[f'W{l}']
            b = self.weights[f'b{l}']
            z = W @ a_prev + b
            
            #Activacion no lineal
            last_mtx = l == L
            if (last_mtx): # caso capa salida
                a = self.softmax(z)
            else: # caso capa oculta
                a = np.maximum(0,z) 
                if self.use_dropout and training:
                    mask = (np.random.rand(*a.shape) >= self.p_dropout).astype(float)
                    a = a * mask / (1 - self.p_dropout)
                    self.cache[f'mask{l}'] = mask
            # Guardo en diccionario
            self.cache[f'z{l}'] = z
            self.cache[f'a{l}'] = a
            
            a_prev = a
        return a_prev
    
    def loss(self, y_pred, y_true):
        eps = 1e-12
        y_true_col = y_true.reshape(-1,1)
        log_probabilities = np.log(y_pred.clip(min=eps)) # (C,1)
        correct_log_prob = np.sum(y_true_col * log_probabilities)
        
        # regularización L2
        L = len(self.mtx_dims)
        l2_reg = 0
        for i in range(1,L+1):
            W = self.weights[f'W{i}']
            l2_reg += np.sum(W**2)
        l2_reg *= self.l2 * 0.5 / self.batch_size
        return -correct_log_prob + l2_reg
    
    def backward(self, x, y_pred, y_true):
        self.deltas.clear()
        L = len(self.mtx_dims) 
        # Última capa
        y_true = y_true.reshape(-1,1)
        deltaL = self.cache[f'a{L}'] - y_true
        dWL = deltaL @ self.cache[f'a{L-1}'].T + self.l2 * self.weights[f'W{L}']
        
        self.deltas[f'delta{L}'] = deltaL
        self.grads[f'dW{L}'] += dWL
        self.grads[f'db{L}'] += deltaL
        
        
        # itero por capas hacia atrás
        for l in reversed(range(1,L)): # recorro hacia atrás
            W_next = self.weights[f'W{l+1}']
            delta_next = self.deltas[f'delta{l+1}']
        
            drelu_curr = self.d_relu(self.cache[f'z{l}'])
            a_prev = self.cache[f'a{l-1}']
            
            delta_curr = W_next.T @ delta_next * drelu_curr
            
            if self.use_dropout and l < L and l > 0:
                delta_curr *= self.cache[f'mask{l}'] / (1 - self.p_dropout)
            dW_curr = delta_curr @ a_prev.T + self.l2 * self.weights[f'W{l}']
            db_curr = delta_curr
            
            self.deltas[f'delta{l}'] = delta_curr
            self.grads[f'dW{l}'] += dW_curr
            self.grads[f'db{l}'] += db_curr
    
    def update_weights(self):
        for i in range(1,len(self.mtx_dims)+1): #Por cada capa
            W = self.weights[f'W{i}']
            b = self.weights[f'b{i}'] 
            dW = self.grads[f'dW{i}']/self.batch_size
            db = self.grads[f'db{i}']/self.batch_size
            lr = self.lr
            if self.optimizer == 'adam':
                self.t += 1
                
                mW = self.m[f'W{i}']
                vW = self.v[f'W{i}']
                mb = self.m[f'b{i}']
                vb = self.v[f'b{i}']
                
                mW = self.beta_1 * mW + (1 - self.beta_1) * dW
                vW = self.beta_2 * vW + (1 - self.beta_2) * dW**2
                mb = self.beta_1 * mb + (1 - self.beta_1) * db
                vb = self.beta_2 * vb + (1 - self.beta_2) * db**2
                
                mW_hat = mW / (1 - self.beta_1**self.t)
                vW_hat = vW / (1 - self.beta_2**self.t)
                mb_hat = mb / (1 - self.beta_1**self.t)
                vb_hat = vb / (1 - self.beta_2**self.t)
                
                W -= lr * mW_hat / (np.sqrt(vW_hat) + self.epsilon)
                b -= lr * mb_hat / (np.sqrt(vb_hat) + self.epsilon)
                self.m[f'W{i}'] = mW
                self.v[f'W{i}'] = vW
                self.m[f'b{i}'] = mb
                self.v[f'b{i}'] = vb
            else: # SGD
                self.weights[f'W{i}'] = W - lr*dW
                self.weights[f'b{i}'] = b - lr*db
            
    def reset_gradients(self):
        for i in range(1, len(self.mtx_dims)+1):
            self.grads[f'dW{i}'].fill(0)
            self.grads[f'db{i}'].fill(0)    
    
    def adam_reset(self):
        self.t = 0
        for i in range(1, len(self.mtx_dims)+1):
            self.m[f'W{i}'].fill(0)
            self.v[f'W{i}'].fill(0)
            self.m[f'b{i}'].fill(0)
            self.v[f'b{i}'].fill(0)
    
    def init_adam(self, beta_1, beta_2, epsilon):
        self.optimizer = 'adam'
        self.t = 0
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        for i in range(1, len(self.mtx_dims)+1):
            self.m[f'W{i}'] = np.zeros_like(self.weights[f'W{i}'])
            self.v[f'W{i}'] = np.zeros_like(self.weights[f'W{i}'])
            self.m[f'b{i}'] = np.zeros_like(self.weights[f'b{i}'])
            self.v[f'b{i}'] = np.zeros_like(self.weights[f'b{i}'])
    
    def d_relu(self, z):
        # z vector de preactivaciones
        return (z > 0).astype(float)
                 
    def softmax(self, z):
        """
        Calcula la función softmax para una matriz de entrada.
        
        Parámetros:
        -----------
        z : np.ndarray
            Matriz de entrada.
        
        Retorna:
        --------
        np.ndarray
            Matriz con la función softmax aplicada.
        """
        exp_z = np.exp(z - np.max(z))
        return exp_z / np.sum(exp_z) 

    """pseudo con agregados
    por cada epoch:
        X = barajar(X)
        batches = dividir(X, batch_size)
        por cada batch:
            gradients = []
            por cada dato:
                y = forward(X)
                loss = compute_loss(y, y_true)
                gradient = backward(X, y, y_true)
                gradients.append(gradient)
            update_weights(gradients)
    """
        