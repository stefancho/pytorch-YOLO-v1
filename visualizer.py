import pandas as pd
import matplotlib.pyplot as plt


class Visualizer():

    def __init__(self, *args, **kwargs):
        """
        :param kwargs: index_column: default is 'epoch'. Depends on whether we are logging for iteration or epoch.
                        file_name: csv file to load logs from(optional)
        """
        self.dataframes = {}
        self.index_column = kwargs.get('index_column', "epoch")
        filename = kwargs.get('file_name', None)
        if filename is not None:
            df = pd.DataFrame.from_csv(filename, index_col=self.index_column)
            self.dataframes = {col: df[[col]].reset_index().dropna() for col in df.columns}

    def add_log(self, data, epoch_num_or_run=None):
        """
        :param epoch_num_or_run:
        :data: tuple (name, value)
        """
        col_name = data[0]
        value = data[1]
        if epoch_num_or_run is None:
            epoch_num_or_run = self._get_last_iteration(col_name) + 1
        if not (col_name in self.dataframes.keys()):
            self.dataframes[col_name] = pd.DataFrame(data={self.index_column: [epoch_num_or_run], col_name: [value]})
        else:
            df = self.dataframes[col_name]

            if not df.loc[df[self.index_column] == epoch_num_or_run].empty:
                raise ValueError("Epoch {} for column {} already logged !".format(epoch_num_or_run, col_name))

            df = df.append(pd.DataFrame(data={self.index_column: [epoch_num_or_run], col_name: [value]}))
            self.dataframes[col_name] = df

    def _get_last_iteration(self, col_name):
        if not (col_name in self.dataframes.keys()):
            return 0
        else:
            df = self.dataframes[col_name]
            return df[self.index_column].iloc[-1]

    def plot(self, columns=None, x_lim=70):
        plt.figure(figsize=(18, 8))
        ax = plt.gca()

        if columns is None:
            columns = self._get_columns()
        df = self._get_dataframe()
        df = df.reset_index()
        for col in columns:
            df.plot(kind='line', x=self.index_column, y=col, ax=ax)
        ax.set_xlim(0, x_lim)
        plt.show()

    def save(self, file_name):
        df = self._get_dataframe()
        df.to_csv(file_name, index=self.index_column)

    def _get_dataframe(self):
        df_list = [df.set_index(self.index_column) for df in self.dataframes.values()]
        return pd.concat(df_list, axis=1)

    def _get_columns(self):
        return [col for col in self.dataframes.keys()]