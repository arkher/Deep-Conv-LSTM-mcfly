import pandas as pd
import sys
from sklearn.model_selection import train_test_split
import gc

CHUNKSIZE = 1e7


def main(argv):
    paths = []

    for i in range(1, len(argv)):
        paths.append(argv[i])

    first_Tr = True
    first_Te = True

    for arch in paths:
        df = pd.read_csv(arch, sep=',')
        gc.collect()
        df.drop(['Processo','Falta_de_Energia', 'Qntd_Processo'], axis=1, inplace=True)

        processos = df.loc[df['CNR'] == 1]
        df = df.loc[df['CNR'] == 0]

        processos = processos.sample(frac=1, random_state=42).reset_index(drop=True)

        # del df
        # #df = df.loc[df['Falta_de_Energia'] == 0]
        # gc.collect()

        # classes = processos.pop('CNR')
        # #aux = df.pop('Falta_de_Energia')
        # #df.insert(0, 'Falta_de_Energia', aux)
        # gc.collect()

        # X_train, X_test, y_train, y_test = train_test_split(
        #     processos, classes, test_size=0.4, random_state=42)
        # gc.collect()

        rowsTr = int(processos.shape[0] * 0.6)
        rowsTe = int(processos.shape[0] * 0.4)

        rowsDF = int(df.shape[0] * 0.6)

        X_train = processos.iloc[:rowsTr, :]
        X_test = processos.iloc[rowsTr:, :]

        #X_train.insert(0, 'CNR', y_train)
        #X_train = pd.concat([X_train, df.loc[:(rowsTr*3), :]])
        X_train = pd.concat([X_train,df.loc[:rowsDF, :]])

        #X_test.insert(0, 'CNR', y_test)
        # X_test = pd.concat([
        #     X_test, df.loc[(rowsTr*3):(rowsTr*3) + (rowsTe*3), :]])
        X_test = pd.concat([X_test, df.loc[rowsDF:, :]])
        if first_Tr is True:
            X_train.to_csv('D:\\cemar_dataset_TRAIN.csv', index=False)
            first_Tr = False
        else:
            with open('D:\\cemar_dataset_TRAIN.csv', 'a') as f:
                X_train.to_csv(f, index=False, header=False)

        if first_Te is True:
            X_test.to_csv('D:\\cemar_dataset_TEST.csv', index=False)
            first_Te = False
        else:
            with open('D:\\cemar_dataset_TEST.csv', 'a') as f:
                X_test.to_csv(f, index=False, header=False)

        print('DONE', arch)
        gc.collect()
        

if __name__ == '__main__':
    main(sys.argv)
