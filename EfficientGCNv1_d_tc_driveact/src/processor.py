import logging, torch, numpy as np
from tqdm import tqdm
from time import time

from . import utils as U
from .initializer import Initializer


class Processor(Initializer):

    def train(self, epoch):
        self.model.train()
        start_train_time = time()
        num_top1, num_sample = 0, 0
        train_iter = self.train_loader if self.no_progress_bar else tqdm(self.train_loader, dynamic_ncols=True)

        y_true = []
        y_pred = []

        for num, (x, y, _) in enumerate(train_iter):
            self.optimizer.zero_grad()

            # Using GPU
            x = x.float().to(self.device)
            y = y.long().to(self.device)
            y_true.extend(list(np.array(y.cpu())))

            # Calculating Output
            out, _ = self.model(x)

            # Updating Weights
            loss = self.loss_func(out, y)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.global_step += 1

            # Calculating Recognition Accuracies
            num_sample += x.size(0)
            reco_top1 = out.max(1)[1]
            y_pred.extend(list(np.array(reco_top1.cpu())))

            num_top1 += reco_top1.eq(y).sum().item()

        # Showing Train Results
        train_acc = num_top1 / num_sample

        from sklearn.metrics import balanced_accuracy_score
        train_balanced_accuracy = balanced_accuracy_score(y_true, y_pred)

        logging.info(
            'Epoch: {}/{}, Training accuracy: {:d}/{:d}({:.2%}), Train balanced_accuracy: {}, Training time: {:.2f}s'.format(
                epoch + 1, self.max_epoch, num_top1, num_sample, train_acc, train_balanced_accuracy,
                time() - start_train_time
            ))

    def eval(self, test=False):
        self.model.eval()

        start_eval_time = time()
        with torch.no_grad():
            num_top1, num_top5 = 0, 0
            num_sample, eval_loss = 0, []
            if test == False:
                eval_iter = self.eval_loader if self.no_progress_bar else tqdm(self.eval_loader, dynamic_ncols=True)
            else:
                eval_iter = self.test_loader if self.no_progress_bar else tqdm(self.test_loader, dynamic_ncols=True)

            y_true = []
            y_pred = []
            score_frag = []

            for num, (x, y, _) in enumerate(eval_iter):

                # Using GPU
                x = x.float().to(self.device)
                y = y.long().to(self.device)
                y_true.extend(list(np.array(y.cpu())))

                # Calculating Output
                out, _ = self.model(x)
                score_frag.append(out.data.cpu().numpy())

                # Getting Loss
                loss = self.loss_func(out, y)
                eval_loss.append(loss.item())

                # Calculating Recognition Accuracies
                num_sample += x.size(0)
                reco_top1 = out.max(1)[1]
                y_pred.extend(list(np.array(reco_top1.cpu())))

                num_top1 += reco_top1.eq(y).sum().item()
                reco_top5 = torch.topk(out, 5)[1]
                num_top5 += sum([y[n] in reco_top5[n, :] for n in range(x.size(0))])

                # Showing Progress
                if self.no_progress_bar and self.args.evaluate:
                    logging.info('Batch: {}/{}'.format(num + 1, len(self.eval_loader)))

        # Showing Evaluating Results
        acc_top1 = num_top1 / num_sample
        acc_top5 = num_top5 / num_sample
        eval_loss = sum(eval_loss) / len(eval_loss)
        eval_time = time() - start_eval_time
        score = np.concatenate(score_frag)
        from sklearn.metrics import recall_score, balanced_accuracy_score
        eval_balanced_accuracy = balanced_accuracy_score(y_true, y_pred)

        eval_speed = len(self.eval_loader) * self.eval_batch_size / eval_time / len(self.args.gpus)
        logging.info(
            'Top-1 accuracy: {:d}/{:d}({:.2%}), Top-5 accuracy: {:d}/{:d}({:.2%}), balanced_accuracy:{:.5}, Mean loss:{:.4f}'.format(
                num_top1, num_sample, acc_top1, num_top5, num_sample, acc_top5, eval_balanced_accuracy, eval_loss
            ))
        logging.info('Evaluating time: {:.2f}s, Speed: {:.2f} sequnces/(second*GPU)'.format(
            eval_time, eval_speed
        ))
        logging.info('')

        torch.cuda.empty_cache()

        return acc_top1, acc_top5, 0, y_true, y_pred, eval_balanced_accuracy, score

    def test_start(self):
        start_time = time()
        best_state = {'acc_top1': 0, 'acc_top5': 0, 'cm': 0, "balanced_acc": 0}
        logging.info('Loading checkpoint ...')
        checkpoint = U.load_checkpoint(self.args.work_dir, self.model_name)
        self.model.module.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        best_state = checkpoint["best_state"]

        logging.info('Best Validation balanced_acc: {:.3%}'.format(best_state['balanced_acc']))
        logging.info('Successful!')
        logging.info('')

        acc_top1, acc_top5, cm, y_true, y_pred, balanced_acc, score = self.eval(test=True)

        np.save(
            "/home/meng/PycharmProjects/EfficientSeries/EfficientGCNv1_d_tc_driveact/resource0527/test0526_{}_{}_{}_data.npy".format(
                self.args.granularity, self.args.datasetCode, self.args.configCode), np.array([y_true, y_pred]))

        logging.info('Best Test balanced_acc: {:.3%},  Total time: {}'.format(
            balanced_acc, U.get_time(time() - start_time)))

    def start(self):
        import random
        import ray
        import numpy as np
        from ray import tune
        from ray.tune import CLIReporter
        from ray.tune.schedulers import ASHAScheduler
        from ray.tune.logger import CSVLoggerCallback
        self.no_progress_bar = True
        start_time = time()
        start_epoch = 0
        best_state = {'acc_top1': 0, 'acc_top5': 0, 'cm': 0, "balanced_acc": 0}
        if self.args.resume:
            logging.info('Loading checkpoint ...')
            checkpoint = U.load_checkpoint(self.args.work_dir)
            self.model.module.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint['epoch']
            best_state.update(checkpoint['best_state'])
            self.global_step = start_epoch * len(self.train_loader)
            logging.info('Start epoch: {}'.format(start_epoch + 1))
            logging.info('Best accuracy: {:.2%}'.format(best_state['acc_top1']))
            logging.info('Successful!')
            logging.info('')

        # Training
        logging.info('Starting training ...')
        for epoch in range(start_epoch, self.max_epoch):

            # Training
            self.train(epoch)

            # Evaluating
            is_best = False
            # logging.info('Evaluating for epoch {}/{} ...'.format(epoch + 1, self.max_epoch))
            acc_top1, acc_top5, cm, y_true, y_pred, balanced_acc, score = self.eval()
            if balanced_acc > best_state['balanced_acc']:
                is_best = True
                best_state.update({'acc_top1': acc_top1, 'acc_top5': acc_top5, 'cm': cm, "balanced_acc": balanced_acc})
                try:
                    np.save(
                        "/home/meng/PycharmProjects/EfficientSeries/EfficientGCNv1_d_tc_driveact/resource0527/valid0526_{}_{}_{}_data.npy".format(
                            self.args.granularity, self.args.datasetCode, self.args.configCode),
                        np.array([y_true, y_pred]))
                except:
                    pass
                # Saving Model
                logging.info('Saving model for epoch {}/{} ...'.format(epoch + 1, self.max_epoch))
                U.save_checkpoint(
                    self.model.module.state_dict(), self.optimizer.state_dict(), self.scheduler.state_dict(),
                    epoch + 1, best_state, is_best, self.args.work_dir, self.save_dir, self.model_name
                )
                logging.info('Best bala_acc: {:.2%},  Total time: {}'.format(
                    best_state['balanced_acc'], U.get_time(time() - start_time)
                ))
                # score_dict = dict(
                #     zip(self.eval_loader.dataset.sample_name, score))
                # basepath = "/home/bullet/PycharmProjects/EfficientSeries/EfficientGCNv1_d_tc_driveact/resources"
                # import pickle
                # with open(basepath + '{}_best_acc'.format(self.args.datasetCode) + '.pkl', 'wb') as f:
                #     pickle.dump(score_dict, f)

            # with tune.checkpoint_dir(epoch) as checkpoint_dir:
            #     import os
            #     path = os.path.join(checkpoint_dir, "checkpoint")
            #     torch.save((self.model.state_dict(), self.optimizer.state_dict()), path)
            # tune.report(loss=(0), accuracy=best_state['acc_top1'], test_acc=acc_top1)
            tune.report(acc_top1=acc_top1, best_bala_acc=best_state['balanced_acc'], balanced_acc=balanced_acc)


            print('Best bala_acc: {:.2%}, Total time: {}'.format(
                best_state['balanced_acc'], U.get_time(time() - start_time)
            ))
            #             logging.info('Best top-1 accuracy: {:.2%},  Total time: {}'.format(
            #                 best_state['acc_top1'], U.get_time(time() - start_time)
            #             ))
            #             print('Best top-1 accuracy: {:.2%}, Total time: {}'.format(
            #                 best_state['acc_top1'], U.get_time(time() - start_time)
            #             ))
            logging.info('')

        logging.info('Finish training!')
        logging.info('')

    def extract(self):
        logging.info('Starting extracting ...')
        if self.args.debug:
            logging.warning('Warning: Using debug setting now!')
            logging.info('')

        # Loading Model
        logging.info('Loading evaluating model ...')
        checkpoint = U.load_checkpoint(self.args.work_dir, self.model_name)
        if checkpoint:
            self.cm = checkpoint['best_state']['cm']
            self.model.module.load_state_dict(checkpoint['model'])
        logging.info('Successful!')
        logging.info('')

        # Loading Data
        x, y, names = iter(self.eval_loader).next()
        location = self.location_loader.load(names) if self.location_loader else []

        # Calculating Output
        self.model.eval()
        out, feature = self.model(x.float().to(self.device))

        # Processing Data
        data, label = x.numpy(), y.numpy()
        out = torch.nn.functional.softmax(out, dim=1).detach().cpu().numpy()
        weight = self.model.module.classifier.fc.weight.squeeze().detach().cpu().numpy()
        feature = feature.detach().cpu().numpy()

        # Saving Data
        if not self.args.debug:
            U.create_folder('./visualization')
            np.savez('./visualization/extraction_{}.npz'.format(self.args.config),
                     data=data, label=label, name=names, out=out, cm=self.cm,
                     feature=feature, weight=weight, location=location
                     )
        logging.info('Finish extracting!')
        logging.info('')
