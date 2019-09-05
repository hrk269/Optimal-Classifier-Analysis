from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import pandas as pd

def knn():
    data = pd.read_csv("./qudditch_training.csv")
    Y = data['quidditch_league_player'].values
    X = data.drop(columns = ['id_num','player_id','gender', 'power_play', 'starfish_and_stick',
             'chelmondiston_charge', 'double_eight_loop', 'finbourgh_flick',
             'plumpton_pass', 'porskoff_ploy', 'transylvanian_tackle',
             'woollongong_shimmy', 'change', 'snitch_caught', 'quidditch_league_player'])

    le = preprocessing.LabelEncoder()
    X['house'] = le.fit_transform(X['house'])
    X['weight'] = le.fit_transform(X['weight'])
    X['player_code'] = le.fit_transform(X['player_code'])
    X['move_specialty'] = le.fit_transform(X['move_specialty'])
    X['player_type'] = le.fit_transform(X['player_type'])
    X['snitchnip'] = le.fit_transform(X['snitchnip'])
    X['stooging'] = le.fit_transform(X['stooging'])
    X['body_blow'] = le.fit_transform(X['body_blow'])
    X['checking'] = le.fit_transform(X['checking'])
    X['dopplebeater_defence'] = le.fit_transform(X['dopplebeater_defence'])
    X['hawkshead_attacking_formation'] = le.fit_transform(X['hawkshead_attacking_formation'])
    X['no_hands_tackle'] = le.fit_transform(X['no_hands_tackle'])
    X['sloth_grip_roll'] = le.fit_transform(X['sloth_grip_roll'])
    X['spiral_dive'] = le.fit_transform(X['spiral_dive'])
    X['twirl'] = le.fit_transform(X['twirl'])
    X['wronski_feint'] = le.fit_transform(X['wronski_feint'])
    X['zig-zag'] = le.fit_transform(X['zig-zag'])
    X['bludger_backbeat'] = le.fit_transform(X['bludger_backbeat'])
    X['dionysus_dive'] = le.fit_transform(X['dionysus_dive'])
    X['reverse_pass'] = le.fit_transform(X['reverse_pass'])
    X['parkins_pincer'] = le.fit_transform(X['parkins_pincer'])
    X['parkins_pincer'] = le.fit_transform(X['parkins_pincer'])

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, stratify = Y)


    knn = KNeighborsClassifier(n_neighbors=100, weights = 'uniform')
    knn.fit(X_train,y_train)

    y_pred = knn.predict(X_test)
    print('Prediction accuracy is {}%'.format(accuracy_score(y_test, y_pred) * 100))

    print(confusion_matrix(y_test, y_pred))
    print(pd.crosstab(y_test, y_pred, rownames=['Correct'], colnames=['Predicted'], margins=True))

def SVM():
    data = pd.read_csv("./qudditch_training.csv")
    Y = data['quidditch_league_player'].values
    X = data[['age', 'foul_type_id', 'game_move_id', 'penalty_id', 'game_duration',
              'num_game_moves', 'num_game_losses', 'num_practice_sessions',
              'num_games_satout', 'num_games_injured', 'num_games_notpartof',
              'num_games_won']]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, stratify = Y)

    model = svm.SVC(kernel='linear')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print('Prediction accuracy is {}%'.format(accuracy_score(y_test, y_pred) * 100))

    print(confusion_matrix(y_test, y_pred))
    print(pd.crosstab(y_test, y_pred, rownames=['Correct'], colnames=['Predicted'], margins=True))


def NN():
    data = pd.read_csv("./qudditch_training.csv")
    Y = data['quidditch_league_player'].values
    X = data[['age', 'foul_type_id', 'game_move_id', 'penalty_id', 'game_duration',
              'num_game_moves', 'num_game_losses', 'num_practice_sessions',
              'num_games_satout', 'num_games_injured', 'num_games_notpartof',
              'num_games_won']]
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, stratify = Y)

    def warn(*args, **kwargs):
        pass
    import warnings
    warnings.warn = warn
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    mlp = MLPClassifier(hidden_layer_sizes=(7), activation = 'relu', solver = 'sgd', learning_rate_init = 0.001, alpha = 0.0001, max_iter=3000)
    mlp.fit(X_train, y_train)

    y_pred = mlp.predict(X_test)
    print('Prediction accuracy is {}%'.format(accuracy_score(y_test, y_pred) * 100))

    print(confusion_matrix(y_test, y_pred))
    print(pd.crosstab(y_test, y_pred, rownames=['Correct'], colnames=['Predicted'], margins=True))


if __name__=="__main__":
    train = pd.read_csv("./qudditch_training.csv")
    y_train = train['quidditch_league_player'].values
    X_train = train[['age', 'foul_type_id', 'game_move_id', 'penalty_id', 'game_duration',
              'num_game_moves', 'num_game_losses', 'num_practice_sessions',
              'num_games_satout', 'num_games_injured', 'num_games_notpartof',
              'num_games_won']]

    test = pd.read_csv("./qudditch_testing_without_target _2_.csv")
    X_test = test[['age', 'foul_type_id', 'game_move_id', 'penalty_id', 'game_duration',
              'num_game_moves', 'num_game_losses', 'num_practice_sessions',
              'num_games_satout', 'num_games_injured', 'num_games_notpartof',
              'num_games_won']]

    output = test[['id_num']]

    def warn(*args, **kwargs):
        pass
    import warnings
    warnings.warn = warn
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    mlp = MLPClassifier(hidden_layer_sizes=(7), activation = 'relu', solver = 'sgd', learning_rate_init = 0.001, alpha = 0.0001, max_iter=3000)
    mlp.fit(X_train, y_train)

    y_pred = mlp.predict(X_test)

    output = pd.concat([output,pd.DataFrame(data = y_pred, columns=['quidditch_league_player'])], axis = 1)
    output = output[['id_num', 'quidditch_league_player']]
    output.to_csv('test_outputs.csv', index=False)
