import logging

# import torchnet as tnt
import sys
import torch

from model.loss_utils import compute_logsumexp_loss, compute_type_loss, compute_ranking_loss
from utils.runner import Runner

logging.basicConfig(format='%(asctime)s: %(filename)s:%(lineno)d: %(message)s', level=logging.INFO)
import numpy as np
import time
from torch.autograd import Variable as V
from torch import FloatTensor as MyTensor


def square(v):
    return v * v


class ELRunner(Runner):
    def __init__(self, args, model, optimizer,
                 maxsteps,
                 test_evaler, train_it, test_it,
                 logfreq, evalfreq, overall=None,
                 dev_it=None, dev_evaler=None,
                 dev2_it=None, dev2_evaler=None,
                 dev3_it=None, dev3_evaler=None):
        super(ELRunner, self).__init__(model, optimizer)
        # self.epoch_loss = tnt.meter.AverageValueMeter()
        self.epoch_loss, self.epoch_cxt_loss = [], []
        if args["usetype"]:
            self.epoch_type_loss = []
        if args["usedesc"]:
            self.epoch_desc_loss = []
        self.args = args
        self.maxsteps = maxsteps
        # self.timer = tnt.meter.TimeMeter(unit=1)
        self.train_it = train_it
        self.test_it = test_it
        self.test_evaler = test_evaler
        self.dev_it = dev_it
        self.dev_evaler = dev_evaler
        self.dev2_it = dev2_it
        self.dev2_evaler = dev2_evaler
        self.dev3_it = dev3_it
        self.dev3_evaler = dev3_evaler
        self.overall = overall
        self.report_after_batches = logfreq
        self.eval_after_batches = evalfreq

    def on_sample(self, state):
        # self.model.hidden = self.model.init_hidden()
        pass

    def on_forward(self, state):
        # self.epoch_loss.add(state['loss'].data[0])
        pass

    def on_start_epoch(self, state):
        self.reset_meters()

    def on_update(self, state, **kwargs):
        pass

    def reset_meters(self):
        self.epoch_start = time.time()
        pass

    def on_end_epoch(self, state):
        elapsed = time.time() - self.epoch_start
        logging.info('epoch:%s Training loss: %.4f time taken:%.2f secs',
                     state['epoch'],
                     np.mean(self.epoch_loss),
                     elapsed)
        print("batches seen", len(self.epoch_loss))
        self.reset_meters()
        self.train_it.reset()

    def get_total_loss(self, logits, batch):
        l_batch, l_lengths, \
        r_batch, r_lengths, \
        truewid_descvec_batch, \
        types_batch, \
        coherence_batch, \
        wids_batch, wid_cprobs_batch, nocands_mask_batch = batch

        cxt_logits = logits["cxt_logits"]
        if self.args["ranking_loss"]:
            cxt_loss = compute_ranking_loss(cxt_logits, ncands=self.args["ncands"])
        else:
            cxt_loss = (0.5 / square(self.model.ecloss_sigma)) * compute_logsumexp_loss(feats_type="cxt",
                                                                                        logits=cxt_logits) \
                       + torch.log(square(self.model.ecloss_sigma))
            # cxt_loss = compute_logsumexp_loss(feats_type="cxt",
            #                                   logits=cxt_logits) 

        if self.args["usedesc"]:
            desc_logits = logits["desc_logits"]
            desc_loss = compute_logsumexp_loss(feats_type="desc", logits=desc_logits)
            self.epoch_desc_loss.append(desc_loss.data[0])
        else:
            desc_loss = self.getZero()

        if self.args["usetype"]:
            type_given_cxt_logits, type_given_ent_logits = logits["type_given_cxt_logits"], logits[
                "type_given_ent_logits"]
            cxt_type_loss = (0.5 / square(self.model.ctloss_sigma)) * compute_type_loss(
                pred_types=type_given_cxt_logits, gold_types=types_batch) \
                            + torch.log(square(self.model.ctloss_sigma))

            ent_type_loss = (0.5 / square(self.model.etloss_sigma)) * compute_type_loss(
                pred_types=type_given_ent_logits,
                gold_types=types_batch) + torch.log(square(self.model.etloss_sigma))
            val = cxt_type_loss.data[0] + ent_type_loss.data[0]
            self.epoch_type_loss.append(val)
        else:
            cxt_type_loss = self.getZero()
            ent_type_loss = self.getZero()

        total_loss = cxt_loss + desc_loss + cxt_type_loss + ent_type_loss

        self.epoch_loss.append(total_loss.data[0])
        self.epoch_cxt_loss.append(cxt_loss.data[0])

        return total_loss

    def get_total_loss2(self, logits, batch):
        l_batch, l_lengths, \
        r_batch, r_lengths, \
        truewid_descvec_batch, \
        types_batch, \
        coherence_batch, \
        wids_batch, wid_cprobs_batch, nocands_mask_batch = batch

        cxt_logits = logits["cxt_logits"]
        if self.args["ranking_loss"]:
            cxt_loss = compute_ranking_loss(cxt_logits, ncands=self.args["ncands"])
        else:
            cxt_loss = compute_logsumexp_loss(feats_type="cxt", logits=cxt_logits)

        if self.args["usedesc"]:
            desc_logits = logits["desc_logits"]
            desc_loss = compute_logsumexp_loss(feats_type="desc", logits=desc_logits)
            self.epoch_desc_loss.append(desc_loss.data[0])
        else:
            desc_loss = self.getZero()

        if self.args["usetype"]:
            type_given_cxt_logits, type_given_ent_logits = logits["type_given_cxt_logits"], logits[
                "type_given_ent_logits"]
            cxt_type_loss = compute_type_loss(pred_types=type_given_cxt_logits,
                                              gold_types=types_batch)

            ent_type_loss = compute_type_loss(pred_types=type_given_ent_logits,
                                              gold_types=types_batch)
            val = cxt_type_loss.data[0] + ent_type_loss.data[0]
            self.epoch_type_loss.append(val)
        else:
            cxt_type_loss = self.getZero()
            ent_type_loss = self.getZero()

        total_loss = cxt_loss + desc_loss + cxt_type_loss + ent_type_loss

        self.epoch_loss.append(total_loss.data[0])
        self.epoch_cxt_loss.append(cxt_loss.data[0])

        return total_loss

    def handle_sample(self, batch, state):
        # if self.args["profile"]: start = time.time()
        batch = self.model.prepare_batch(batch)
        # if self.args["profile"]: end = time.time()
        # if self.args["profile"]: logging.info("time spent prepping batch %.4f secs", end - start)
        # if self.args["profile"]: start = time.time()

        logits = self.model.forward(batch)
        # if self.args["profile"]: end = time.time()
        # if self.args["profile"]: logging.info("time in forward pass %.4f secs", end - start)

        if state['t'] > 0 and state['t'] % self.report_after_batches == 0:
            logging.info("seen %s batches", state['t'])
            logging.info("ecsigma %s etsigma %s ctsigma %s", self.model.ecloss_sigma.data[0], self.model.etloss_sigma.data[0], self.model.ctloss_sigma.data[0])
            loss_log = "loss:%.3f " % np.mean(self.epoch_loss[-50:])
            loss_log += "cxt:%.3f " % np.mean(self.epoch_cxt_loss[-50:])
            if self.args["usetype"]:
                loss_log += "typ:%.3f " % np.mean(self.epoch_type_loss[-50:])
            if self.args["usedesc"]:
                loss_log += "des:%.3f " % np.mean(self.epoch_desc_loss[-50:])
            logging.info(loss_log)

        if state['t'] % self.eval_after_batches == 0:
            model_test_score, cxt_test_score = self.test_evaler.do_eval(args=self.args,
                                                                        runner=self,
                                                                        model=self.model,
                                                                        test_iterator=self.test_it,
                                                                        batch_idx=state['t'])
            logging.info("test acc: %.4f", model_test_score)
            logging.info("best mod:%.4f at %d cxt:%.4f at %d",
                         self.test_evaler.get_best(),
                         self.test_evaler.get_best_batch(),
                         self.test_evaler.get_cxt_best(),
                         self.test_evaler.get_cxt_best_batch())
            if self.dev_it is not None:
                model_dev_score, cxt_dev_score = self.dev_evaler.do_eval(args=self.args,
                                                                         runner=self,
                                                                         model=self.model,
                                                                         test_iterator=self.dev_it,
                                                                         batch_idx=state['t'])
                logging.info("dev acc: %.4f", model_dev_score)
                logging.info("best mod:%.4f at %d cxt:%.4f at %d",
                             self.dev_evaler.get_best(),
                             self.dev_evaler.get_best_batch(),
                             self.dev_evaler.get_cxt_best(),
                             self.dev_evaler.get_cxt_best_batch())
            if self.dev2_it is not None:
                model_dev2_score, cxt_dev2_score = self.dev2_evaler.do_eval(args=self.args,
                                                                            runner=self,
                                                                            model=self.model,
                                                                            test_iterator=self.dev2_it,
                                                                            batch_idx=state['t'])
                logging.info("dev acc: %.4f", model_dev2_score)
                logging.info("best mod:%.4f at %d cxt:%.4f at %d",
                             self.dev2_evaler.get_best(),
                             self.dev2_evaler.get_best_batch(),
                             self.dev2_evaler.get_cxt_best(),
                             self.dev2_evaler.get_cxt_best_batch())
            if self.dev3_it is not None:
                model_dev3_score, cxt_dev3_score = self.dev3_evaler.do_eval(args=self.args,
                                                                            runner=self,
                                                                            model=self.model,
                                                                            test_iterator=self.dev3_it,
                                                                            batch_idx=state['t'])
                logging.info("dev acc: %.4f", model_dev3_score)
                logging.info("best mod:%.4f at %d cxt:%.4f at %d",
                             self.dev3_evaler.get_best(),
                             self.dev3_evaler.get_best_batch(),
                             self.dev3_evaler.get_cxt_best(),
                             self.dev3_evaler.get_cxt_best_batch())

            overall_scores = model_test_score
            if self.dev_it is not None:
                overall_scores = (model_dev_score, model_test_score)
            if self.dev2_it is not None:
                overall_scores = (model_dev2_score, model_dev_score, model_test_score)
            if self.dev3_it is not None:
                overall_scores = (model_dev3_score, model_dev2_score, model_dev_score, model_test_score)
            logging.info("overall %s", overall_scores)
            self.overall.update(batch_idx=state['t'],
                                scores=overall_scores)
            # if self.args["save"] and self.overall.best_was_updated:
            #     ELRunner.save_checkpoint({
            #         'args': self.args,
            #         'state_dict': self.model.state_dict(),
            #         'best_scores': self.overall.best_scores,
            #         'optimizer': self.optimizer.state_dict(),
            #     }, is_best=True, filename=self.args["save"])
            logging.info("overall best %s %d", self.overall.best_scores, self.overall.best_batch)

        if state['t'] >= self.maxsteps:
            logging.info("enough batches seen, stopping.")
            if self.args["save"]:
                ELRunner.save_checkpoint({
                    'args': self.args,
                    'state_dict': self.model.state_dict(),
                    'best_scores': self.overall.best_scores,
                    'optimizer': self.optimizer.state_dict(),
                }, is_best=True, filename=self.args["save"])
            logging.info("overall %s %d", self.overall.best_scores, self.overall.best_batch)
            sys.exit(0)

        output = logits["cxt_logits"]
        loss = self.get_total_loss(logits=logits, batch=batch)

        return loss, output

    def on_end(self, state):
        pass

    def getZero(self):
        if self.args["device_id"] is not None:
            return V(MyTensor([0.0])).cuda(device=self.args["device_id"])
        else:
            return V(MyTensor([0.0]))
