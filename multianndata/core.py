import anndata as ad
import numpy as np
import pandas as pd

class MultiAnnData(ad.AnnData):
    def __init__(self, *args, samplem=None, sampleid='id', **kwargs):
        def verify_numeric(df, name):
            non_numeric = df.columns.values[[not pd.api.types.is_numeric_dtype(df[c])
                                for c in df.columns]]
            if len(non_numeric) > 0:
                print('warning: the following columns of '+name+' are non-numeric.')
                print(non_numeric)
                print('consider casting to numeric types where appropriate, and')
                print('consider re-coding text-valued columns with pandas.get_dummies')

        self.sampleid = sampleid
        super().__init__(*args, **kwargs)
        if samplem is not None:
            self.samplem = samplem
            verify_numeric(self.samplem, 'samplem')
        elif self.samplem is None and self.obs_sampleids is not None:
            self.samplem = pd.DataFrame(
                index=pd.Series(self.obs_sampleids.unique(), name=self.sampleid))
            verify_numeric(self.samplem, 'samplem')
            verify_numeric(self.obs, 'obs')
        else:
            self._check()

    def _check(self):
        if self.samplem is None:
            print('warning: samplem is absent')
            return
        if self.obs_sampleids is None:
            print('warning: per-observation sample ids not found')
            return
        if not set(self.obs_sampleids.unique()).issubset(set(self.sampleids)):
            print('warning: there are observations with unrecognized sample ids')

    @property
    def samplem(self):
        return self.uns['sampleXmeta'] if 'sampleXmeta' in self.uns else None
    @samplem.setter
    def samplem(self, value):
        self.uns['sampleXmeta'] = value
        self._check()
    @samplem.deleter
    def samplem(self):
        del self.uns['sampleXmeta']

    @property
    def N(self):
        return len(self.samplem) if self.samplem is not None else None

    @property
    def sampleid(self):
        return self._sampleid
    @sampleid.setter
    def sampleid(self, value):
        self._sampleid = value

    @property
    def sampleids(self):
        return self.samplem.index

    @property
    def obs_sampleids(self):
        return self.obs[self.sampleid] if self.sampleid in self.obs.columns else None

    @property
    def sample_sizes(self):
        return self.obs[self.sampleid].value_counts()

    def obs_to_sample(self, columns, aggregate=np.mean):
        if type(columns) == str:
            columns = [columns]
        for c in columns:
            self.samplem.loc[:,c] = \
                self.obs[[self.sampleid, c]].groupby(by=self.sampleid).aggregate(aggregate)

    def merge_duplicates(self, sample_info):
        self.obs[self.sampleid] = self.obs[self.sampleid].replace(
                            to_replace=sample_info.index,
                            value=sample_info.values)
        self.samplem[sample_info.name] = sample_info
        new_samplem = self.samplem.drop_duplicates(subset=sample_info.name)
        if len(new_samplem) != len(self.samplem.drop_duplicates()):
            print('warning: samples with non-identical covariates are being merged')
            print('you may not want to merge these samples')
        self.samplem = new_samplem.set_index(
                        sample_info.name, drop=True)
