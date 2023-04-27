import cv2
import numpy as np

from sklearn.neighbors import DistanceMetric
from collections import defaultdict

class Confusions(object):
    @staticmethod
    def collect(true, pred, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def allowed_average_types():
        return set()

class SegmentationConfusions(Confusions):
    def collect(true, pred, **kwargs):
        assert true.shape == pred.shape

        confusions = defaultdict(dict)

        if kwargs.get('nclasses', None) is None:
            nclasses = np.unique(true)
        elif isinstance(nclasses, int):
            nclasses = np.arange(nclasses)
        else:
            raise ValueError(f'Incorrect type - {type(nclasses)} of nclasses variable. Type must be int or None')

        for label in nclasses:
            selected_true = (true == label)
            selected_pred = (pred == label)

            confusions[label]['TP'] = np.sum((selected_true == selected_pred) & selected_true)
            confusions[label]['FP'] = np.sum((selected_true != selected_pred) & np.logical_not(selected_true))
            confusions[label]['FN'] = np.sum((selected_true != selected_pred) & np.logical_not(selected_pred))

        return confusions

    def allowed_average_types():
        return { 'micro', 'macro', 'none', 'binary' }

class SeedDetectionConfusions(Confusions):
    """ Metric from article "Deep Learning for Multi-task Plant Phenotyping"
    """

    @staticmethod
    def _get_central_points(mask):
        contours, _ = cv2.findContours( mask,
                                        cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_NONE )

        central_points = list()

        for contour in contours:
            try:
                moments = cv2.moments(contour)

                cx = int(moments['m10']/moments['m00'])
                cy = int(moments['m01']/moments['m00'])

                central_points.append([cx, cy])
            except:
                pass

        return central_points

    @staticmethod
    def _compute_confusion(points, pred_points, max_distance):
        if len(pred_points) == 0:
            return { 'TP': 0, 'FP': 0, 'FN': len(points) }

        euclidean = DistanceMetric.get_metric('euclidean')
        distances = euclidean.pairwise(points, pred_points)

        pairs = list()

        for tidx, _ in enumerate(points):
            for pidx, _ in enumerate(pred_points):
                pairs.append((distances[tidx][pidx], tidx, pidx))

        sorted_by_distance_pairs = sorted(pairs, key=lambda x:x[0])

        correct_pairs = defaultdict(list)
        distributed_pred = set()

        for distance, tidx, pidx in sorted_by_distance_pairs:
            if distance > max_distance:
                break
            if pidx not in distributed_pred:
                correct_pairs[tidx].append(pidx)
                distributed_pred.add(pidx)

        confusion = { 'TP': 0, 'FP': 0,  'FN': 0 }        
        confusion['FN'] = len(points) - len(correct_pairs)

        for neighbors in correct_pairs.values():
            if len(neighbors) > 1:
                confusion['FP'] += len(neighbors) - 1

            confusion['TP'] += 1
            
        for i in range(len(pred_points)):
            if i not in distributed_pred:
                confusion['FP'] += 1
        return confusion

    def collect(true, pred, **kwargs):
        assert 'distance' in kwargs, 'Not found distance in kwargs'

        points = SeedDetectionConfusions._get_central_points(true)
        pred_points = SeedDetectionConfusions._get_central_points(pred)

        confusions = defaultdict(dict)
        confusions[1] = SeedDetectionConfusions._compute_confusion(points, pred_points, kwargs['distance'])

        return confusions

    def allowed_average_types():
        return { 'binary' }
