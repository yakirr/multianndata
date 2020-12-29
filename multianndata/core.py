import anndata as ad
import numpy as np

class MultiAnnData(ad.AnnData):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampleid = 'id'
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
