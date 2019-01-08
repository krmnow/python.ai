import numpy as np

class Environment(object):
    
    def __init__(self, optimal_temperature = (18.0, 24.0), initial_month = 0, initial_number_users = 10, initial_rate_data = 60):
        self.monthly_atmospheric_temperatures = [1.0, 5.0, 7.0, 10.0, 11.0, 20.0, 23.0, 24.0, 22.0, 10.0, 5.0, 1.0]
        self.initial_month = initial_month
        self.atmospheric_temperatures = self.monthly_atmospheric_temperatures[initial_month]
        self.optimal_temperature = optimal_temperature
        self.min_temperature = -20
        self.max_temperature = 80
        self.min_numer_users = 10
        self.max_number_users = 100
        self.max_update_users = 5
        self.min_rate_data = 20
        self.max_rate_data = 300
        self.max_update_data = 10
        self.initial_number_users = initial_number_users
        self.current_number_users = initial_number_users
        self.initial_rate_data = initial_rate_data
        self.current_rate_data = initial_rate_data
        self.intrinsic_temperature = self.atmospheric_temperatures + 1.25 * self.current_number_users + 1.25 * self.current_rate_data
        self.temperature.ai = self.intrinsic_temperature
        self.temperature_noai = (self.optimal_temperature[0] + self.optimal_temperature[1]) / 2.0
        self.total_energy_ai = 0.0
        self.total_energy_noai = 0.0
        self.reward = 0.0
        self.game_over = 0
        self.train = 1
        
    def update_env(self, direction, energy_ai, month):
        #GETTING THE REWARD
        
        #Comouting the energy spent by server' scooling system when there is no ai
         
        energy_noai = 0
            if self.temperature_noqai < self.optimal_temperature[0]:
                energy_noai = self.optimal_temperature[0] - self.temperature_noqai
                self.temperature_noai = self.optimal_temperature[0]
            elif self.temperature_noqai > self.optimal_temperature[1]:
                energy_noai = self.temperature_noqai - self.optimal_temperature[1]
                #cpmputing the reward
            self.reward = energy_noai - ebergu_ai
            #scaling the reward
            self.reward = le-3 * self.reward
            
            #GETTING THE NEXT STATE
            
            #Updating the atmospheric temperature
            self.atmospheric_temperatures = monthly_atmospheric_temperatures[month]
            #updating the numbers if users
            self.current_number_users += np.random.randint(-self.max_update_users, self.max_update_users) 
            if self.current_number_users > self.max_number_users:
                self.current_number_users = self.max_number_users
            elif self.current_number_users < self.max_number_users:
                self.current_number_users = self.min_number_users
            #updating the rate of data
           self.current_number_users += np.random.randint(-self.max_update_users, self.max_update_users) 
            if self.current_rate_data > self.max_rate_data:
                self.current_rate_data = self.max_rate_data
            elif self.current_rate_data < self.max_rate_data:
                self.current_rate_data = self.min_rate_data 
            #computing the delta of intrinsic temperature
            past_intrinsic_temperature = self.intrinsic_temperature
            self.intrinsic_temperature = self.atmospheric_temperatures + 1.25 * self.current_number_users + 1.25 * self.current_rate_data
            delta_ = self.intrinsic_temperature = self.intrinsic_temperature - past_intrinsic_temperature
            #computing the delta of temperature caused by AI
            if direction == -1:
                delta_detmeprature_ai = -energy_ai
            elif direction == 1:
                delta_detmeprature_ai = energy_ai
            #updating the new serwr temperature caused by AI
            self.temperature_ai += delta_intrinsic_temperature + delta_temperature_ai
            #updating the new serwer temperature when there is no AI
            self.temperature_noai += delta_intrinsic_temperature
            
            #GETTING GAME OVER
            if self.temperature_ai < self.min_temperature:
                if self.train == 1:
                    self.game_over = 1
                else:
                    self.temperature_ai = self.optimal_temperature[0]
                    self.total_energy_ai += self.optimal_temperature[0] - self.temperature_ai
            elif self.temperature_ai > self.max_temperature:
                if self.train == 1:
                    self.game_over = 1
                else:
                    self.temperature_ai = self.optimal_temperature[1]
                    self.total_ebergy_ai += self.temperature_ai - self.optimal_temperature[1]
                    
            #UPDATING THE SCORES
            
            #updating the total energy spent by the AI
            self.total_energy_ai += energy_ai
            #updating the total energy spent by serwer's cooling system when there is no AI
            self.total_energy_ai += energy_noai  
            
            #SCALING THE NEXT STAGE
            
            scaled_temperature_ai = (self.temperature_ai - self.min_temperature) / (self.max_temperature - self.min_temperature)
            scaled_number_users = (self.current_number_users - self.min_number_users) / (self.max_number_users - self.min_number_users)
            scaled_rate_data = (self.current_rate_data - self.min_rate_data) / (self.max_rate_data - self.min_rate_data)
            next_state = np.matrix([scaled_temperature_ai, scaled_number_users, scaled_rate_data])
            
            #RETURNING THE NEXT STATE, THE REWARD AND GAME OVER
            return next_state, self.reward, self.game_over
#MAKING A METHOD THAT RESETS THE ENVOIRMENT

    def reset(self, new_month):
        self.atmospheric_temperatures = self.monthly_atmospheric_temperatures[NEW_month]
        self.initial_mont = new_month
        self.current_rate_data = self.initial_rate_data
        self.current_number_users = self.initial_number_users
        self.intrinsic_temperature = self.atmospheric_temperatures + 1.25 * self.current_number_users + 1.25 * self.current_rate_data
        self.temperature.ai = self.intrinsic_temperature
        self.temperature_noai = (self.optimal_temperature[0] + self.optimal_temperature[1]) / 2.0
        self.total_energy_ai = 0.0
        self.total_energy_noai = 0.0
        self.reward = 0.0
        self.game_over = 0
        self.train = 1
        
    #MAKING A METHOD THAT GIVES US AT ANY TIME THE CURRENY STATE, THE LAST REWARD AND WHETEWE THE GME IS OVER
        
    def observe(self):
        scaled_temperature_ai = (self.temperature_ai - self.min_temperature) / (self.max_temperature - self.min_temperature)
        scaled_number_users = (self.current_number_users - self.min_number_users) / (self.max_number_users - self.min_number_users)
        scaled_rate_data = (self.current_rate_data - self.min_rate_data) / (self.max_rate_data - self.min_rate_data)
        current_state = np.matrix([scaled_temperature_ai, scaled_number_users, scaled_rate_data])
        return current/-state, self.reward, self.game_over
