import numpy
import prettytable
import scipy
from scipy.optimize import linear_sum_assignment

_EPSILON = 1e-10


class PanopticQuality:

    def __init__(self, num_categories, is_thing_mask, min_size_stuff=2048, ignored_category=None):

        # boolean array length num_categories
        self.num_categories = num_categories
        self.is_thing = numpy.array(is_thing_mask)
        self.num_thing = numpy.sum(is_thing_mask)
        self.num_stuff = numpy.sum(~is_thing_mask)
        self.min_size_stuff = min_size_stuff

        self.stuff_weighting = self.num_stuff / self.num_categories
        self.thing_weighting = self.num_thing / self.num_categories

        # Ignored background class
        self.ignored_category = ignored_category

        # per-image scores
        self.per_im_scores = {
            "All": {
                "PQ": [],
                "RQ": [],
                "SQ": []
            },
            "Things": {
                "PQ": [],
                "RQ": [],
                "SQ": []
            },
            "Stuff": {
                "PQ": [],
                "RQ": [],
                "SQ": []
            }
        }

        # accumulators per class
        self.iou_per_class = numpy.zeros(self.num_categories, dtype=numpy.float64)
        self.tp_per_class = numpy.zeros(self.num_categories, dtype=numpy.float64)
        self.fn_per_class = numpy.zeros(self.num_categories, dtype=numpy.float64)
        self.fp_per_class = numpy.zeros(self.num_categories, dtype=numpy.float64)

    def compare_and_accumulate(self, gt_sem, gt_inst, gt_inst_labels, pred_sem, pred_inst, pred_inst_labels):
        """

        :param gt_sem: numpy array (n_classes, h, w, d) containing ground truth semantic segmentations
        :param gt_inst: numpy array (n_instances, h, w, d) containing ground truth instance masks
        :param gt_inst_labels: numpy array ( n_instances) containing the class labels of the
        ground truth  instances. Ground truth labels start from 0. Thing classes have own labels
        :param pred_sem: numpy array ( n_classes, h, w, d) containing predicted semantic segmentations
        :param pred_inst: numpy array (n_instances, h, w, d) containing predicted instance masks
        :param pred_inst_labels: numpy array (n_instances) containing the class labels of the
        predicted instances. Ground truth labels start from 0. Thing classes have own labels
        :return:
        """

        # Stuff Part
        sem_iou_img = numpy.zeros(self.num_categories)
        sem_iou_mask = numpy.full_like(sem_iou_img, False)

        sem_tp = 0
        sem_fn_fp = 0

        # Calculate semantic iou
        stuff_classes = numpy.where(~self.is_thing)[0]
        for idx in stuff_classes:
            has_gt = numpy.sum(gt_sem[idx] > 0.5) > self.min_size_stuff
            has_pred = numpy.sum(pred_sem[idx] > 0.5) > self.min_size_stuff
            # If correct yes / no
            has_matched = has_gt == has_pred
            sem_iou_mask[idx] = has_matched

            if has_matched is True:
                sem_tp += 1
                self.tp_per_class[idx] += 1
            else:
                sem_fn_fp += 1
                # Doesnt matter for the formula
                self.fn_per_class[idx] += 1

            class_iou = self.compute_iou(gt_sem[idx], pred_sem[idx])
            sem_iou_img[idx] = class_iou
            self.iou_per_class[idx] += class_iou

        sem_sq_img = numpy.mean(sem_iou_img[sem_iou_mask])
        sem_rq_img = 2 * sem_tp / (2 * sem_tp + sem_fn_fp)
        sem_pq_img = sem_rq_img

        self.per_im_scores["Stuff"]["PQ"].append(sem_pq_img)
        self.per_im_scores["Stuff"]["SQ"].append(sem_sq_img)
        self.per_im_scores["Stuff"]["RQ"].append(sem_rq_img)

        # Thing Part
        sem_iou_things = numpy.zeros(self.num_categories)
        num_gt_inst = gt_inst.shape[0]
        num_pred_inst = pred_inst.shape[0]

        gt_labels, gt_counts = numpy.unique(gt_inst_labels, return_counts=True)

        sq_class_things = []
        rq_class_things = []
        pq_class_things = []

        inst_matches = numpy.zeros((num_gt_inst, num_pred_inst))

        for idx_gt in range(num_gt_inst):
            for idx_pred in range(num_pred_inst):
                if gt_inst_labels[idx_gt] != pred_inst_labels[idx_pred]:
                    continue
                inst_matches[idx_gt, idx_pred] = self.compute_iou(gt_inst[idx_gt], pred_inst[idx_pred])

        for c in gt_labels:

            gt_label_match = gt_inst_labels == c
            pred_label_match = pred_inst_labels == c

            sum_mask_gt = numpy.sum(gt_inst[gt_label_match], axis=0) > 0
            sum_mask_pred = numpy.sum(pred_inst[pred_label_match], axis=0) > 0

            sem_iou_things[c] = self.compute_iou(sum_mask_gt, sum_mask_pred)

            gt_c = numpy.sum(gt_label_match)
            pred_c = numpy.sum(pred_label_match)

            match_mat = inst_matches[gt_label_match][:, pred_label_match]
            # Optimal matching

            matched_gt, matched_pred = linear_sum_assignment(-match_mat)  # maximize IoU
            matched = []

            for i, j in zip(matched_gt, matched_pred):
                iou = match_mat[i, j]
                if iou > 0.5:  # PQ matching threshold
                    self.iou_per_class[c] += iou
                    self.tp_per_class[c] += 1
                    matched.append((i, j))

            fn_c_img = gt_c - len(matched)
            fp_c_img = pred_c - len(matched)

            self.fn_per_class[c] += fn_c_img
            self.fp_per_class[c] += fp_c_img

            # Recognition Quality: F1-score of the class
            class_rq = 2 * len(matched) / (fp_c_img + fn_c_img + 2 * len(matched))
            class_sq = sem_iou_things[c]
            class_pq = class_sq * class_rq

            sq_class_things.append(class_sq)
            rq_class_things.append(class_rq)
            pq_class_things.append(class_pq)

        # Class-wise-accumulation
        inst_pq_img = numpy.mean(pq_class_things)
        inst_sq_img = numpy.mean(sq_class_things)
        inst_rq_img = numpy.mean(rq_class_things)

        self.per_im_scores["Things"]["PQ"].append(inst_pq_img)
        self.per_im_scores["Things"]["SQ"].append(inst_sq_img)
        self.per_im_scores["Things"]["RQ"].append(inst_rq_img)

        self.per_im_scores["All"]["PQ"].append(
            sem_pq_img * self.stuff_weighting + inst_pq_img * self.thing_weighting)
        self.per_im_scores["All"]["SQ"].append(
            sem_sq_img * self.stuff_weighting + inst_sq_img * self.thing_weighting)
        self.per_im_scores["All"]["RQ"].append(
            sem_rq_img * self.stuff_weighting + inst_rq_img * self.thing_weighting)

    def _valid_categories(self):
        valid = numpy.not_equal(self.tp_per_class + self.fn_per_class + self.fp_per_class, 0)
        if 0 <= self.ignored_label < self.num_categories:
            valid[self.ignored_label] = False
        return valid

    def compute_iou(self, mask1, mask2):
        intersection = numpy.logical_and(mask1, mask2).sum()
        union = numpy.logical_or(mask1, mask2).sum()
        return self.realdiv_maybe_zero(intersection, union)

    def detailed_results(self):
        valid = self._valid_categories()
        sq = self.realdiv_maybe_zero(self.iou_per_class, self.tp_per_class)
        rq = self.realdiv_maybe_zero(self.tp_per_class,
                                     self.tp_per_class + 0.5*self.fn_per_class + 0.5*self.fp_per_class)
        pq = sq*rq
        res = {}
        res['All'] = {
            'pq': numpy.mean(pq[valid]),
            'sq': numpy.mean(sq[valid]),
            'rq': numpy.mean(rq[valid]),
            'n': numpy.sum(valid)
        }
        # Things
        thing_mask = numpy.logical_and(valid, self.is_thing)
        res['Things'] = {
            'pq': numpy.mean(pq[thing_mask]) if thing_mask.any() else 0,
            'sq': numpy.mean(sq[thing_mask]) if thing_mask.any() else 0,
            'rq': numpy.mean(rq[thing_mask]) if thing_mask.any() else 0,
            'n': numpy.sum(thing_mask)
        }
        # Stuff
        stuff_mask = numpy.logical_and(valid, ~self.is_thing)
        res['Stuff'] = {
            'pq': numpy.mean(pq[stuff_mask]) if stuff_mask.any() else 0,
            'sq': numpy.mean(sq[stuff_mask]) if stuff_mask.any() else 0,
            'rq': numpy.mean(rq[stuff_mask]) if stuff_mask.any() else 0,
            'n': numpy.sum(stuff_mask)
        }
        return res

    def result_with_confidence_interval(self, scores, confidence=0.95):
        n = len(scores)
        if n <= 1:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        loo = numpy.array([(sum(scores) - scores[i]) / (n - 1) for i in range(n)])
        mean = loo.mean()
        stderr = loo.std(ddof=1) / numpy.sqrt(n)
        z = scipy.stats.t.ppf(1 - (1 - confidence) / 2, df=n - 1)
        margin = z * stderr
        return mean, stderr, mean - margin, mean + margin, margin

    def calculate_margins(self):
        return {
            'All': self.result_with_confidence_interval(numpy.array(self.per_im_scores["All"]["PQ"])),
            'Things': self.result_with_confidence_interval(numpy.array(self.per_im_scores["Things"]["PQ"])),
            'Stuff': self.result_with_confidence_interval(numpy.array(self.per_im_scores["Stuff"]["PQ"]))
        }

    def calculate_margins_rq(self):
        return {
            'All': self.result_with_confidence_interval(numpy.array(self.per_im_scores["All"]["RQ"])),
            'Things': self.result_with_confidence_interval(numpy.array(self.per_im_scores["Things"]["RQ"])),
            'Stuff': self.result_with_confidence_interval(numpy.array(self.per_im_scores["Stuff"]["RQ"]))
        }

    def calculate_margins_sq(self):
        return {
            'All': self.result_with_confidence_interval(numpy.array(self.per_im_scores["All"]["SQ"])),
            'Things': self.result_with_confidence_interval(numpy.array(self.per_im_scores["Things"]["SQ"])),
            'Stuff': self.result_with_confidence_interval(numpy.array(self.per_im_scores["Stuff"]["SQ"]))
        }

    def print_detailed_results_with_ci(self, print_digits=3):
        res = self.detailed_results()
        margins = self.calculate_margins()
        sq_margins = self.calculate_margins_sq()
        rq_margins = self.calculate_margins_rq()

        tab = prettytable.PrettyTable()
        tab.field_names = ['Set', 'PQ', 'SQ', 'RQ', 'N', 'PQ ± CI', 'SQ ± CI', 'RQ ± CI']
        for key in ['All', 'Things', 'Stuff']:
            pq = round(res[key]['pq'], print_digits)
            sq = round(res[key]['sq'], print_digits)
            rq = round(res[key]['rq'], print_digits)
            n = res[key]['n']
            mean, stderr, low, high, margin = margins[key]
            mean_sq, _, _, _, margin_sq = sq_margins[key]
            mean_rq, _, _, _, margin_rq = rq_margins[key]
            tab.add_row([key, pq, sq, rq, n,
                         f"{round(mean,print_digits)} ± {margin:.{print_digits}e}",
                         f"{round(mean_sq,print_digits)} ± {margin_sq:.{print_digits}e}",
                         f"{round(mean_rq,print_digits)} ± {margin_rq:.{print_digits}e}"])
        print(tab)
        return tab.__str__()

    def result(self):
        return numpy.mean(self.per_im_scores["All"]["PQ"])

    def reset(self):
        for key in self.per_im_scores:
            for metric in self.per_im_scores[key]:
                self.per_im_scores[key][metric].clear()
                self.iou_per_class.fill(0)
                self.tp_per_class.fill(0)
                self.fn_per_class.fill(0)
                self.fp_per_class.fill(0)

    def realdiv_maybe_zero(self, x, y):
        return numpy.where(numpy.less(numpy.abs(y), _EPSILON), numpy.zeros_like(x), numpy.divide(x, y))

    @classmethod
    def preprocess_for_panoptic_quality(cls, pred_inst, pred_sem, annot_inst, annot_sem, stuff_classes):
        """

        :param pred_inst: Predicted Instance masks as InstanceList
        :param pred_sem: Predicted Semantic segmentation as np array [b, n, h, w, d]
        :param annot_inst:
        :param annot_sem:
        :param stuff_classes:
        :param void_value:
        :return:
        """

        # unchanged implementation
        shape = annot_sem.shape[1:]

        # Handle the semantic segmentation => Convert to binary masks
        sem_areas_annot = numpy.zeros((stuff_classes, shape[0], shape[1], shape[2]))
        sem_areas_pred = numpy.zeros_like(sem_areas_annot)
        for idx in range(pred_sem.shape[0]):
            sem_areas_pred = numpy.where(pred_sem[idx] > 0.5, idx, sem_areas_pred)
        for idx in range(annot_sem.shape[0]):
            sem_areas_annot = numpy.where(annot_sem[idx] > 0.5, idx, sem_areas_annot)

        # Get instance masks an labels
        if len(pred_inst.pred_masks) > 0:
            pred_instances_array = pred_inst.pred_masks.tensor.cpu().numpy()
            pred_label_array = pred_inst.get("pred_classes").detach().cpu().numpy()
        # Handle the case that there are no instance masks
        else:
            pred_instances_array = numpy.zeros((0, *shape))
            pred_label_array = numpy.zeros((0,))

        annot_inst_array = numpy.zeros((len(annot_inst), shape[0], shape[1], shape[2]))
        annot_label_array = numpy.zeros((len(annot_inst), ))
        for idx, inst in enumerate(annot_inst.items()):
            annot_inst_array[idx] = inst[1]["mask"]
            annot_label_array[idx] = inst[1]["label"].item()

        return sem_areas_annot, annot_inst_array, sem_areas_pred, pred_instances_array, annot_label_array, \
            pred_label_array
