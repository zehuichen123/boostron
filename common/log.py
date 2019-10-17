import nni

class LogHelper():
    def __init__(self, nni_log):
        self.nni_log = nni_log

    def kfold_single(self, idx, train_loss, val_loss):
        print("[Fold {}] train loss:{}, vali loss:{}".format(idx, train_loss, val_loss))
        if self.nni_log == True:
            nni.report_intermediate_result(val_loss)

    def kfold_summary(self, summary_loss):
        print("[Folds Summary] train loss:{}".format(summary_loss))
        if self.nni_log == True:
            nni.report_final_result(summary_loss)