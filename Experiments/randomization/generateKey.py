import numpy as np
import random
import pandas as pd

saveFileAs = 'randomization_key.csv'

# Participants
participants = ['HC{:02d}'.format(i+1) for i in range(30)] + ['Test01']

# Simulation conditions
c_control  = ['CAM']
c_sharpnet = ['SN3','SN5']
c_canny    = ['CE%d' %(i+1) for i in range(6)]
c_simulation = c_control + c_canny + c_sharpnet

# Environmental conditions
c_difficulty = ['simple', 'complex'] # difficulty
c_route = ['route_%d' %(i+1) for i in range(14)] # route

# Output table
randomization = pd.DataFrame(columns = ['seed','participant','session','trial','condition_1','condition_2','condition_3'])

# For each participant (with corresponding random seed) generate a randomized trial protocol
for seed,participant in enumerate(participants):

    #### Test session ####
    session = 0
    trial = 0
    model = 'canny'
    for i, simulation in enumerate(['CAM','CAM','CE6','CE6','SN5','SN5','SN3','SN3','CE2','CE2']):
        trial = trial + 1
        route = 'route_0'
        difficulty = c_difficulty[(trial+1)%2]
        randomization = randomization.append(pd.Series({'seed':seed,
                                                        'participant':participant,
                                                        'session':session,
                                                        'trial':trial,
                                                        'condition_1':simulation,
                                                        'condition_2':difficulty,
                                                        'condition_3':route}),ignore_index=True)


    ### Measurement sessions ###
    # 2 sessions * (1 camera condition + 2*6 Canny conditions + 2*2 SharpNet conditions)
    random.seed(seed)
    for session in [1,2]:

        # Random permutations
        key1 = [i for i in range(7)] # Route key
        key2 = [i+1 for i in range(8)] # order key a
        key3 = [i+1 for i in range(8)] # order key b
        random.shuffle(key1)
        random.shuffle(key2)
        random.shuffle(key3)
        key2 = [0]+key2
        key3 = [0]+key3

        # Phosphenes are randomly shuffled
        simulation_simple = [c_simulation[key2[i]] for i in range(9)]
        simulation_complex = [c_simulation[key2[key3[i]]] for i in range(9)]

        # Routes are mirrored (+7) for the same phosphene condition and equal for the SN and corresponding CE condition
        key2[key2.index(7)] = 3
        key2[key2.index(8)] = 5
        routes_simple = [c_route[key1[key2[i]]] for i in range(9)]
        routes_complex = [c_route[7+key1[key2[key3[i]]]] for i in range(9)]


        for i in range(9):
            for difficulty in c_difficulty:
                trial = trial + 1
                simulation = {'simple':simulation_simple,'complex':simulation_complex}[difficulty][i]
                route = {'simple':routes_simple,'complex':routes_complex}[difficulty][i]
                randomization = randomization.append(pd.Series({'seed':seed,
                                                                'participant':participant,
                                                                'session':session,
                                                                'trial':trial,
                                                                'condition_1':simulation,
                                                                'condition_2':difficulty,
                                                                'condition_3':route,}),ignore_index=True)

randomization.to_csv(saveFileAs,index=False)

### Un-comment to save randomization keys for individual participants
# for participant in participants:
#     part_randomization = randomization.loc[randomization.participant==participant]
#     part_randomization.to_csv('%s_randomization.csv' %participant,index=False)
