import os
import gym
import pathlib
from stable_baselines3 import A2C as A2C_
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike

from config.default import RLConfig


class A2C:
    def __init__(self, id_, config=None, pretrained_model=None):
        # Define um ID para a Instância/Simulacao
        self._id = id_

        if config is None:
            self.config = RLConfig()
        else:
            self.config = config

        # Define qual a arquitetura da rede neural interna da DQN
        self.policy = dict(net_arch=self.config.net_arch,
                           optimizer_class=RMSpropTFLike,
                           optimizer_kwargs=dict(eps=1e-5))
        # self.policy = dict(optimizer_class=RMSpropTFLike,
        #                    optimizer_kwargs=dict(eps=1e-5))

        # Instancia o ambiente Gym Customizado para o Problema
        self.env = gym.make('gym_phishing:RLPTraining-v0')

        # Define o total de timesteps
        self.total_timesteps = self.config.total_timesteps

        # Instancia o Modelo
        if pretrained_model is None:
            self.model = A2C_("MlpPolicy",
                              self.env,
                              policy_kwargs=self.policy,
                              learning_rate=self.config.learning_rate,
                              verbose=self.config.verbose)
        # Ou carrega um modelo treinado anteriormente
        else:
            self.model = A2C_.load(pretrained_model)

    def train(self):
        self.model.learn(total_timesteps=self.total_timesteps)
        full_path = os.path.join(pathlib.Path().resolve(), "models", "A2Cmodel_{}.zip".format(self._id))
        self.model.save(full_path)

    def test(self):
        self.env = gym.make('gym_phishing:RLPTest-v0')
        obs = self.env.reset()
        self.model.set_env(self.env)
        contador = 0
        for _ in range(1000):
            action, _states = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = self.env.step(action)
            self.env.render()
            if info['acertou']:
                contador += 1
            if done:
                obs = self.env.reset()
        print("Acurácia A2C: {}".format(contador / 1000))
        return contador / 1000
