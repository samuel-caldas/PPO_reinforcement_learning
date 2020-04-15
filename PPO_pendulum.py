"""
A simple version of Proximal Policy Optimization (PPO)
Based on:
1. [https://arxiv.org/abs/1707.02286]
2. [https://arxiv.org/abs/1707.06347]
View more on this tutorial website: https://morvanzhou.github.io/tutorials
and this video: https://www.youtube.com/watch?v=lehLSoMPmcM&t=144s
"""
#   Importaçoes   #####################################################################################################################################
#import tensorflow as tf
# Compatibilidade com tensorflow v1
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt # Plota graficos matematicos
import gym  # Environment: ambiente onde a simulação vai acontecer

#   Configurações   ###################################################################################################################################

EP_MAX = 600            # Qantidade total de episódios
EP_LEN = 200            # Quantas sequencias vão acontecer dentro de cada episódio
GAMMA = 0.9             # Avanço (?)
A_LR = 0.0001           # Taxa de aprendizado do ATOR
C_LR = 0.0002           # Taxa de aprendizado da CRITICA
BATCH = 64              # Tamanho do pacote à entrar para treinamento em cada etapa (?)
A_UPDATE_STEPS = 20     # Quantidade de vezes que o treinamento do ATOR vai tomar a cadeia de dados de batch
C_UPDATE_STEPS = 20     # Quantidade de vezes que o treinamento da CRITICA vai tomar a cadeia de dados de batch
S_DIM, A_DIM = 3, 1     # S_DIM é a dimensao do estado, ou seja, quantas entradas ele terá
                        # A_DIM é a dimensão das ações, ou seja, quantas acões podem ser executadas

METHOD = dict(name='clip', epsilon=0.2)     # Metodo de clip sujerido pelos papéis como mais eficiente
                                            # (Clipped surrogate objective)
                                            # epsilon=0.2 Valor de epsilon sujerido pelos papéis

#   Implementaçao da classe ppo   #####################################################################################################################

class PPO(object):  # Classe PPO agrega:
                    #   As redes neurais ATOR e CRITICA;
                    #   Função para atualizar as redes neurais;
                    #   Função para obter o valor de aprendizagem da CRITICA;
                    #   Função para treinar as redes neurais;
                    #   Função para escolher uma ação;

    def __init__(self): # Inicializador da Classe
        self.sess = tf.Session()    #inicializar uma seção do TensorFlow
        # Declaração das entradas das redes:
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM ], 'state')  # Estado do ambiente: a rede recebe o estado do ambiente através desse placeholder
        self.tfa = tf.placeholder(tf.float32, [None, A_DIM], 'action')  # Ação escolhida pela rede é informada através desse placeholder 
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage') # Calculo do ganho que a rede obteve no episódio, calculado fora da classe PPO.
                                                                        # Necessário para treinar tanto o ATOR quanto a CRITICA

        # CRITICA:
        with tf.variable_scope('critic'):   # Criação da rede neural
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu, name='layer1-critic')   # Camada 1 entrada da Critica: 
                                                                                    #   self.tfs é o placeholder do estado, funciona como entrada da rede
                                                                                    #   100 é o numero de neuronios 
                                                                                    #   Relu é o tipo de ativação da saida da rede
            self.v = tf.layers.dense(l1, 1, name = 'V_layer')                       # Camada de saida de valores da CRITICA: 
                                                                                    #   l1 é a variavel referente a primeira camada da rede, 
                                                                                    #   1 é a quantidade de saidas da rede
                                                                                    #   A saida dessa rede será o Q-Value, o status do progreço do aprendizado
        # Metodo de treinamento para o CRITICA, ou seja, o metodo de aprendizagem:
        with tf.variable_scope('ctrain'):
            self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')     # A recompensa de cada episódio é inserida na rede através desse placeholder
            self.advantage = self.tfdc_r - self.v                                   # Atraves da recompensa discounted_r/tfdc_r subtraida pelo valor de aprendizagem V_layer/v obtemos a vantagem
            self.closs = tf.reduce_mean(tf.square(self.advantage))                  # tf.square calcula o quadrado da vantagem e tf.reduce_mean calcula a média. 
                                                                                    # ! Através disso obtemos em closs o Loss ou a Perda da CRITICA
            self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)      # Ultilizamos o otimizador ADAM, com a taxa de aprendizado da CRITICA C_LR
                                                                                    # com a funçao minimize processamos os gradientes da CRITICA através da perda da CRITICA em closs
                                                                                    #   Poderiamos usar tambem o SGD como otimizador.

        # ATOR:
        #   Politica atual
        pi, pi_params = self._build_anet('pi', trainable=True)                  # Criação da rede neural (pi) para a politica atual do ATOR através da função build_anet, definindo como treinavel
                                                                                #   pi é a saida da rede e pi_params são os pesos (estado atual) da rede
                                                                                #   Os pesos pi_params sao ultilizados para atualizar as politicas atual a antiga.

        with tf.variable_scope('sample_action'):
            self.sample_op = tf.squeeze(pi.sample(1), axis=0)   # Tira uma amostra de açao da politica atual pi do ATOR

        #   Politica antiga
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)        # Criação da rede neural oldpi para a politica antiga do ATOR através da função build_anet, definindo como não treinavel

        with tf.variable_scope('update_oldpi'):                                                 # Atualização dos pesos dos pesos de oldpi tendo como referencia os pesos de pi
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)] # A cada atualização da rede, os pesos da politica atual passam para a politica antiga
                                                                                                # Update_oldpi_op acumula todos os valores de pi ao decorrer do episodio

        # Implementação da função de perda PPO
        with tf.variable_scope('loss'): # Funçao de perda:
            with tf.variable_scope('surrogate_pp'):
                ratio = pi.prob(self.tfa) / oldpi.prob(self.tfa)    # O Ratio é a razão da probabilidade da ação tfa na politica nova 
                                                                    # pela probabilidade da ação tfa na politica antiga.
                surr = ratio * self.tfadv                           # Surrogate é a Razão multiplicada pela vantagem

            self.aloss = -tf.reduce_mean(                           # tf.educe_mean calcula a negativa da média do
                tf.minimum(                                         #   menor valor entre
                    surr,                                           #       o Surrogate e
                    self.tfadv*                                     #       a multiplicação da vantagem
                        tf.clip_by_value(                           #           pelo ratio clipado (limitado) por
                            ratio,                                  #               
                            1.-METHOD['epsilon'],                   #                no maximo 1 - o metodo Clipped surrogate objective
                            1.+METHOD['epsilon']                    #                no minimo 1 + o metodo Clipped surrogate objective              
                        )                                           # 
                )                                                   # Obtendo assim em aloss a perda do Ator
            )

        # Metodo de treinamento para o ATOR, ou seja, o metodo de aprendizagem:
        with tf.variable_scope('atrain'):
            self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)  # Ultilizamos o otimizador ADAM, com a taxa de aprendizado do ATOR A_LR
                                                                                # com minimize processamos os gradientes do ATOR através da perda do ATOR em aloss

        tf.summary.FileWriter("log/", self.sess.graph)      # Salvando o modelo na pasta log para analize futura no tensorboard

        self.sess.run(tf.global_variables_initializer())    # Inicializando todas as váriaveis definidas
    
    # Função de atualizaçao
    def update(self, s, a, r): # Recebe o estado, a ação e a recompensa
        self.sess.run(self.update_oldpi_op) # Executa a matriz update_oldpi_op que comtem todos os pesos de pi/oldpi
        
        # Atualiza o ATOR
        adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})  # Calcula a vantagem, ou seja, a recompensa do ATOR através 
        [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(A_UPDATE_STEPS)] # Treina o ator

        # Atualiza a CRITICA através da função de treinamento
        [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(C_UPDATE_STEPS)]   # Executa o treinamento da critica
                                                                                                        #   ctrain_op é o treinamento da critica
                                                                                                        #   tfs é o placehoder que recebe estado s do ambiente
                                                                                                        #   tfdc_r é o placeholder que recebe a recompensa r do ambiente
                                                                                                        #   C_UPDATE_STEPS é quantas vezes que a rede vai ser atualizada

    def _build_anet(self, name, trainable): # Constroi as redes neurais do ATOR 
                                            #   name é o nome da rede
                                            #   trainable determina se a rede é treinavel ou nao 
                                            #   Build the current & hold structure for the policies 
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu, trainable=trainable)    # Camada 1 entrada do ATOR: 
                                                                                    #   self.tfs é o placeholder do estado, funciona como entrada pra rede
                                                                                    #   100 é o numero de neuronios 
                                                                                    #   Relu é o tipo de ativação da saida da rede
                                                                                    #   trainable determina se a rede é treinavel ou nao
            mu = 2 * tf.layers.dense(l1, A_DIM, tf.nn.tanh, trainable=trainable, name = 'mu_'+name) # Camada mu do ATOR
                                                                                                    #   Calcula a ação que vai ser tomada
                                                                                                    #   l1 é a entrada da camada
                                                                                                    #   A_DIM
                                                                                                    #   tanh é o tipo de ativação da saida da camada, retorna um valor entre 1 e -1, 
                                                                                                    #   que está sendo multiplicado por 2 para se adequar ao ambiente.
                                                                                                    #   trainable determina se a rede é treinavel ou nao
            sigma = tf.layers.dense(l1, A_DIM, tf.nn.softplus, trainable=trainable,name ='sigma_'+name )    # Camada sigma do ATOR
                                                                                                            #   Calcula o desvio padrão, o range onde estará a possibilidade de ação 
                                                                                                            #   l1 é a entrada da camada
                                                                                                            #   A_DIM
                                                                                                            #   softplus é o tipo de ativação da saida da camada 
                                                                                                            #   trainable determina se a rede é treinavel ou nao
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)            # Normaliza a saida mu da rede, considerando sigma
                                                                                # Loc é a média
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)   # Coleta em params os pesos das camadas l1,mu/2 e sigma
        return norm_dist, params    # Retorna a ação e os pesos atuais das redes para serem armazenados na politica antiga.

    def choose_action(self, s):     # Recebe o estado s e retorna uma ação a
        s = s[np.newaxis, :]        #   Recebe o estado s e 
        a = self.sess.run(self.sample_op, {self.tfs: s})[0] #   Executa sample_op com o placeholder tfs que receve o estado s e armazena a açao em a
        return np.clip(a, -2, 2)    #   Retorna um valor de ação a clipado entre -2 e 2
                                    #   Limita la salida de valores entre -2 & 2, a cada uno de los valores de 'a'

    def get_v(self, s):             # Recebe o estado s e retorna o valor da taxa de aprendizagem da CRITICA
        if s.ndim < 2: s = s[np.newaxis, :] # 
        return self.sess.run(self.v, {self.tfs: s})[0, 0]   # Retorna a taxa de aprendizagem da CRITICA
                                                            # v é a saida de valores da CRITICA
                                                            # tfs é o placeholder que recebe o estado s
                                                            # Salida de NN del Critic|| V = learned state-value function

#   Implementaçao do ambiente   ########################################################################################################################

env = gym.make('Pendulum-v0').unwrapped # Instancia o ambiente pendulo
ppo = PPO()                             # Instancia a classe PPO
all_ep_r = []                           # Cria um array para a recompensa de todos os episodios

#   Loop de episódios   #
for ep in range(EP_MAX):                # EP_MAX: quantidade de episodios 
    s = env.reset()                     # Redefine o ambiente e armazena o estado atual em s
    buffer_s, buffer_a, buffer_r = [], [], []   # Cria tres arrais para o episódio:
                                                #   buffer_s: buffer do estado
                                                #   buffer_a: buffer da ação
                                                #   buffer_r: buffer da recompensa
    ep_r = 0                            # Recompensa do episódio
#   Loop de episódio    #    
    for t in range(EP_LEN):             # Duração de cada episodio
        env.render()                    # Renderiza o ambiente
        a = ppo.choose_action(s)        # Envia um estado s e recebe uma açao a 
        s_, r, done, _ = env.step(a)    # Envia uma açao a ao ambiente e recebe o estado s_, e a recompensa r
                                        # observation, reward, done, info|| 'a' is torque
        buffer_s.append(s)              # Adiciona ao buffer de estado o estado atual s
        buffer_a.append(a)              # Adiciona ao buffer de ação a açao atual a
        buffer_r.append((r+8)/8)        # Adiciona ao buffer de recompensa a recompensa atual (?) normalizada (r+8)/8
                                        # Normalize reward, find to be useful
        #print(r)
        s = s_      # Atualiza a variavel de estado com o estado recebido pelo ambiente
        ep_r += r   # soma a recompensa da ação a recompensa do episodio

        # update ppo
        if (t+1) % BATCH == 0 or t == EP_LEN-1: #
            v_s_ = ppo.get_v(s_)    # Passa o estado atual s_ e recebe o valor atual da taxa de aprendizagem da CRITICA
                                    # Obteniendo la respuesta de la NN del Critic, entregando el estado 's_' 
                                    # V = learned state-value function
            discounted_r = []       # Cria um array pra armazenar as recompensas calculadas
            for r in buffer_r[::-1]:
                v_s_ = r + GAMMA * v_s_     # Calcula a recompensa multiplicando a recompensa recebida r pela GAMMA 
                                            # e pelo valor da taxa de aprendizado do estado v_s_
                discounted_r.append(v_s_)   # Adiciona ao array de recompensas calculadas 
            discounted_r.reverse()          # Coloca o array de recompensas calculadas ao contrario

            bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]    # vstack trnasforma os arrays que estão em linha, em colunas
                                                                                                            # Esses arrays de colunas sao armazenados em bs ba e br
            buffer_s, buffer_a, buffer_r = [], [], []   # Esvazia os buffers de estado, açao e recompensa
            ppo.update(bs, ba, br)                      # Atualiza as redes com:
                                                        #   Os estados aculmulados
                                                        #   As ações aculmuladas
                                                        #   As recompensas aculmuladas
                                                        # Treine o cliente e o ator (status, ações, desconto de r)
            
    if ep == 0: all_ep_r.append(ep_r)
    else: all_ep_r.append(all_ep_r[-1]*0.9 + ep_r*0.1)
    print(
        'Ep: %i' % ep,
        "|Ep_r: %i" % ep_r,
        ("|Lam: %.4f" % METHOD['lam']) if METHOD['name'] == 'kl_pen' else '',
    )

plt.plot(np.arange(len(all_ep_r)), all_ep_r)
plt.xlabel('Episode');plt.ylabel('Moving averaged episode reward');plt.show()
