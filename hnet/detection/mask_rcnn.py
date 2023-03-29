from .utils_det import *
import torchvision.models.detection as tmdet
# from torchvision.models.detection.rpn import *
# from torchvision.models.detection.roi_heads import *


class RegionProposalNetwork(tmdet.rpn.RegionProposalNetwork):
    def compute_losses(self, outputs, targets=None):
        """ Compute the RPN Loss.
        Args:
            outputs: 
                proposals (List[Tensor]): the predicted boxes from the RPN, one Tensor per image.
                scores,
                anchors,
                objectness,
                pred_bbox_deltas,
            targets (List[Dict[Tensor]]): ground-truth boxes present in the image (optional).
                    If provided, each element in the dict should contain a field `boxes`,
                    with the locations of the ground-truth boxes.
        Returns:
            losses (Dict[Tensor]): the losses for the model during training. During testing, it is an empty dict.
        """
        anchors = outputs['anchors']
        objectness = outputs['objectness']
        pred_bbox_deltas = outputs['pred_bbox_deltas']
        
        losses = {}
        if self.training:
            assert targets is not None
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
            loss_objectness, loss_rpn_box_reg = super().compute_loss(
                objectness, pred_bbox_deltas, labels, regression_targets)
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg,
            }
        
        return losses
        
    def forward(self, features, task_size):
        """ Remove dependencies on images (ImageList). 
        Args:
            features (OrderedDict[Tensor]): {'0': (batch, n_channel, h, w)}
            task_size (Tuple[int, int]): (roi_h, roi_w)
        Returns:
            boxes (List[Tensor]): the predicted boxes from the RPN, one Tensor per image.
            scores (List[Tensor]): the prediceted scores for each box.
            objectness: 
            pred_bbox_deltas: 
            anchors:
        """
        # RPN uses all feature maps that are available
        features = list(features.values())
        objectness, pred_bbox_deltas = self.head(features)
        # Original code use task_size = image_list.tensors.shape[-2:], 
        # Here we only generate anchor inside task_roi
        anchors = self.anchor_generator(task_size, features)

        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
        objectness, pred_bbox_deltas = \
            tmdet.rpn.concat_box_prediction_layers(objectness, pred_bbox_deltas)
        # apply pred_bbox_deltas to anchors to obtain the decoded proposals note that 
        # we detach the deltas because Faster R-CNN do not backprop through the proposals
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)
        # Original code use image_shapes=images.image_sizes to crop proposals
        # We constrain proposals inside task_roi
        image_shapes = [task_size for _ in range(num_images)]
        boxes, scores = self.filter_proposals(proposals, objectness, image_shapes, num_anchors_per_level)
        
        return {'proposals': boxes, 'scores': scores, 'anchors': anchors,
                'objectness': objectness, 'pred_bbox_deltas': pred_bbox_deltas}


## copy from tmdet.roi_heads.RoIHeads, class_logits, masks_logits
class RoIHeads(tmdet.roi_heads.RoIHeads):
#     def compute_class_loss(self, class_logits, box_regression, targets):
#         for t in targets:
#             # TODO: https://github.com/pytorch/pytorch/issues/26731
#             floating_point_types = (torch.float, torch.double, torch.half)
#             assert t["boxes"].dtype in floating_point_types, 'target boxes must of float type'
#             assert t["labels"].dtype == torch.int64, 'target labels must of int64 type'
#             if self.has_keypoint():
#                 assert t["keypoints"].dtype == torch.float32, 'target keypoints must of float type'

#         proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        
        
#         if self.training:
#             assert labels is not None and regression_targets is not None
#             loss_classifier, loss_box_reg = fastrcnn_loss(
#                 class_logits, box_regression, labels, regression_targets)
#             losses = {
#                 "loss_classifier": loss_classifier,
#                 "loss_box_reg": loss_box_reg,
#             }
    
    def postprocess_train_proposals(self, class_logits, box_regression, proposals, image_shapes, 
                                    matched_idxs=None, labels=None, regression_targets=None):
        """ Keep all boxes for all classes. Leave NMS etc in post-processing. 
            class_logits: [n_image*n_obj_per_image, n_classes], flattened class logits for selected proposals
            box_regression: [n_image*n_obj_per_image, n_classes * 4], flattened box for each class for selected proposals
            proposals: [n_obj_per_image, 4] * n_image, selected proposals (pos vs. neg)
            image_shapes: List[Tuple[int, int]]
            
            matched_idxs (Optional): [n_obj_per_image] * n_image, matched obj_id in gt
            labels (Optional): [n_obj_per_image] * n_image, 0 means negative, matched label in gt
            regression_targets (Optional): [n_obj_per_image, 4] * n_image, matched bbox in gt
        """
        ## pack class_logits and box_regression with image_id
        device = class_logits.device
        num_classes = class_logits.shape[-1]
        num_images = len(proposals)
        
        objects: Dict[str, List[torch.Tensor]] = {}
        objects["proposals"] = [p for p in proposals]
        boxes_per_image = [p.shape[0] for p in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals).split(boxes_per_image, 0)
        pred_boxes = [torchvision.ops.boxes.clip_boxes_to_image(boxes, image_shape)
                      for boxes, image_shape in zip(pred_boxes, image_shapes)]
        objects["boxes"] = pred_boxes
        objects["logits"] = [logit for logit in class_logits.split(boxes_per_image, 0)]
        # objects["labels"] = [logit.argmax(-1) for logit in objects["logits"]]
        # pred_probs = F.softmax(class_logits, -1)
        
        if self.training: 
            assert matched_idxs is not None
            assert labels is not None
            assert regression_targets is not None
            objects["matched_idxs"] = list(matched_idxs)
            objects["gt_labels"] = list(labels)
            objects["gt_boxes"] = list(regression_targets)
            
            ## save postivie bbox, class logits, matched_idxs 
            for img_id in range(num_images):
                pos = torch.where(objects["gt_labels"][img_id] > 0)[0]
                for k in objects:
                    objects[k][img_id] = objects[k][img_id][pos]

        return objects
    
    def forward(self, features, proposals, image_shapes, targets=None):
        """
        Args:
            features (Dict[str, Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets Optional[List[Dict[str, Tensor]]]
        """
        if targets is not None:
            for t in targets:
                # TODO: https://github.com/pytorch/pytorch/issues/26731
                floating_point_types = (torch.float, torch.double, torch.half)
                assert t["boxes"].dtype in floating_point_types, 'target boxes must of float type'
                assert t["labels"].dtype == torch.int64, 'target labels must of int64 type'
                if self.has_keypoint():
                    assert t["keypoints"].dtype == torch.float32, 'target keypoints must of float type'

        if targets is not None:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            matched_idxs, labels, regression_targets = None, None, None
        
        # proposals: [n_proposal, 4] * n_image
        # box_features: [N_image*n_proposal, c, roi_output_size[0], roi_output_size[1]]
        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)  # [N_image*n_proposal, 1024]
        class_logits, box_regression = self.box_predictor(box_features)
        
        result: List[Dict[str, torch.Tensor]] = []
        losses = {}
        if targets is not None:
            assert labels is not None and regression_targets is not None
            loss_classifier, loss_box_reg = tmdet.roi_heads.fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets)
            losses = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg,
            }
            ## record positive predictions
            assert matched_idxs is not None
            objects = self.postprocess_train_proposals(
                class_logits, box_regression, proposals, image_shapes, 
                matched_idxs, labels, regression_targets)
            for i in range(len(proposals)):
                result.append({k: objects[k][i] for k in objects})
        else:
            ## apply mask filtering in postprocess_detection
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append({"boxes": boxes[i], "labels": labels[i], "scores": scores[i],})

        if self.has_mask():
            if targets is not None:
                assert matched_idxs is not None
                # during training, only focus on positive boxes
                mask_proposals = [p["proposals"] for p in result]
                pos_matched_idxs = [p["matched_idxs"] for p in result]
            else:
                mask_proposals = [p["boxes"] for p in result]
                pos_matched_idxs = None

            if self.mask_roi_pool is not None:
                mask_features = self.mask_roi_pool(features, mask_proposals, image_shapes)
                mask_features = self.mask_head(mask_features)
                mask_logits = self.mask_predictor(mask_features)
            else:
                raise Exception("Expected mask_roi_pool to be not None")

#                 boxes_per_image = [p["boxes"].shape[0] for p in result]
#                 mask_boxes_per_class = [p["boxes"].flatten(start_dim=0, end_dim=1) for p in result]
#                 mask_logits = torch.cat([logit[i % num_classes] for i, logit in enumerate(mask_logits)])
            
            loss_mask = {}
            if targets is not None:
                assert targets is not None
                assert pos_matched_idxs is not None
                assert mask_logits is not None

                gt_masks = [t["masks"] for t in targets]
                gt_labels = [t["labels"] for t in targets]
                rcnn_loss_mask = tmdet.roi_heads.maskrcnn_loss(
                    mask_logits, mask_proposals,
                    gt_masks, gt_labels, pos_matched_idxs)
                loss_mask = {
                    "loss_mask": rcnn_loss_mask
                }
                
                discretization_size = mask_logits.shape[-1]
                pos_gt_labels = [r["gt_labels"] for r in result]
                pos_gt_masks = [
                    tmdet.roi_heads.project_masks_on_boxes(m, p, i, discretization_size)
                    for m, p, i in zip(gt_masks, mask_proposals, pos_matched_idxs)
                ]
                # masks_probs = maskrcnn_inference(mask_logits, pos_gt_labels)
                # The following code will keep all mask for each class: [N_obj, n_classes, 28, 28]
                boxes_per_image = [len(_) for _ in pos_matched_idxs]
                masks_probs = mask_logits.sigmoid().split(boxes_per_image, dim=0)
                
                for mask_prob, gt_mask, r in zip(masks_probs, pos_gt_masks, result):
                    r["masks"] = mask_prob
                    r["gt_mask"] = gt_mask
            else:
                masks_probs = tmdet.roi_heads.maskrcnn_inference(mask_logits, labels)
                # The following code will keep all mask for each class: [N_obj, n_classes, 28, 28]
                # boxes_per_image = [len(_) for _ in pos_matched_idxs]
                # masks_probs = mask_logits.sigmoid().split(boxes_per_image, dim=0)
                for mask_prob, r in zip(masks_probs, result):
                    r["masks"] = mask_prob
            
            losses.update(loss_mask)

        # keep none checks in if conditional so torchscript will conditionally
        # compile each branch
        if self.keypoint_roi_pool is not None and self.keypoint_head is not None \
                and self.keypoint_predictor is not None:
            if self.training:
                assert matched_idxs is not None
                # during training, only focus on positive boxes
                keypoint_proposals = [p["proposals"] for p in result]
                pos_matched_idxs = [p["matched_idxs"] for p in result]
            else:
                keypoint_proposals = [p["boxes"] for p in result]
                pos_matched_idxs = None

            keypoint_features = self.keypoint_roi_pool(features, keypoint_proposals, image_shapes)
            keypoint_features = self.keypoint_head(keypoint_features)
            keypoint_logits = self.keypoint_predictor(keypoint_features)

            loss_keypoint = {}
            if self.training:
                assert targets is not None
                assert pos_matched_idxs is not None

                gt_keypoints = [t["keypoints"] for t in targets]
                rcnn_loss_keypoint = tmdet.roi_heads.keypointrcnn_loss(
                    keypoint_logits, keypoint_proposals,
                    gt_keypoints, pos_matched_idxs)
                loss_keypoint = {
                    "loss_keypoint": rcnn_loss_keypoint
                }
                ## TODO: add keypoint in training here
            else:
                assert keypoint_logits is not None
                assert keypoint_proposals is not None

                keypoints_probs, kp_scores = tmdet.roi_heads.keypointrcnn_inference(keypoint_logits, keypoint_proposals)
                for keypoint_prob, kps, r in zip(keypoints_probs, kp_scores, result):
                    r["keypoints"] = keypoint_prob
                    r["keypoints_scores"] = kps

            losses.update(loss_keypoint)

        return result, losses


class MaskRCNN(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        # build connection module, match roi_size into target_size for best resolution
        # Assume 20x input has each patch 256x256, but mask is labeled under 40x, target_size is 512x512,
        # connector will enhance feature_maps resolution 2x for small object detections.
        stride_h = int(math.log2(config['roi_size'][0]/config['target_size'][0]))
        stride_w = int(math.log2(config['roi_size'][1]/config['target_size'][1]))
        assert stride_h == stride_w, "Different stride is not supported yet!"
        stride = min(stride_h, stride_w)
        self.connector = DetectionFeatureConnector(
            config['in_channels'], config['in_channels'], stride, config['feature_maps'], mode='upsample',
        )
        
        ## build proposal extraction module
        self.config = get_rcnn_config(config, num_classes=config['num_classes'], masks=False, keypoints=None)
        self.config['featmap_names'] = list(self.config['feature_maps'].keys())
        self.rpn = self.get_rpn()
        
        ## build predictor module
        self.roi_heads = self.get_roi_heads()
        # used only on torchscript mode
        self._has_warned = False
        
        # load_pretrain(self, pretrained, layers="headers")
    
    def save_config(self, filepath):
        with open(filepath, 'w') as f:
            yaml.safe_dump(self.config, f, default_flow_style=False)
    
    def get_rpn(self):
        in_channels = self.config['in_channels']
        rpn_params = self.config['rpn_params']
        
        rpn_anchor = AnchorGenerator(**rpn_params['anchor'])
        rpn_header = tmdet.rpn.RPNHead(in_channels, rpn_anchor.num_anchors_per_location()[0])
        rpn = RegionProposalNetwork(rpn_anchor, rpn_header, **rpn_params['rpn'])
        
        return rpn
    
    def get_roi_heads(self):
        featmap_names = self.config['featmap_names']
        in_channels = self.config['in_channels']
        roi_params = self.config['roi_params']
        
        ## box header
        box_header = BoxPredictor(in_channels, featmap_names, **roi_params['box'])
        
        ## roi heads, write tmdet.roi_heads.RoIHeads
        roi_heads = RoIHeads(
            box_roi_pool=box_header.box_roi_pool,
            box_head=box_header.box_head,
            box_predictor=box_header.box_predictor,
            **roi_params['roi']
        )
        
        ## add mask header
        if 'mask' in roi_params and roi_params['mask'] is not None:
            mask_header = MaskPredictor(in_channels, featmap_names, **roi_params['mask'])
            roi_heads.mask_roi_pool = mask_header.mask_roi_pool
            roi_heads.mask_head = mask_header.mask_head
            roi_heads.mask_predictor = mask_header.mask_predictor
        
        ## add keypoint header
        if 'keypoint' in roi_params and roi_params['keypoint'] is not None:
            keypoint_header = KeypointPredictor(in_channels, featmap_names, **roi_params['keypoint'])
            roi_heads.keypoint_roi_pool = keypoint_header.keypoint_roi_pool
            roi_heads.keypoint_head = keypoint_header.keypoint_head
            roi_heads.keypoint_predictor = keypoint_header.keypoint_predictor
        
        return roi_heads
    
    def postprocess(self, result, image_sizes, original_image_sizes=None):
        # result: List[Dict[str, Tensor]]
        # image_sizes: List[Tuple[int, int]]
        # original_image_size: List[Tuple[int, int]]
        if self.training:
            return result
        original_image_sizes = original_image_sizes or image_sizes
        for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_sizes, original_image_sizes)):            
            boxes = pred["boxes"]
            boxes = resize_boxes(boxes, im_s, o_im_s)
            result[i]["boxes"] = boxes
            if "masks" in pred:
                masks = pred["masks"]
                masks = paste_masks_in_image(masks, boxes, o_im_s)
                result[i]["masks"] = masks
            if "keypoints" in pred:
                keypoints = pred["keypoints"]
                keypoints = resize_keypoints(keypoints, im_s, o_im_s)
                result[i]["keypoints"] = keypoints
        return result
    
    def remove_degenerate_boxes(self, targets):
        for target_idx, target in enumerate(targets):
            boxes = target["boxes"]
            # Check boxes shape
            if isinstance(boxes, torch.Tensor):
                if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                    raise ValueError("Expected target boxes to be a tensor"
                                     "of shape [N, 4], got {:}.".format(
                                         boxes.shape))
            else:
                raise ValueError("Expected target boxes to be of type "
                                 "Tensor, got {:}.".format(type(boxes)))
            
            # Check degenerate boxes
            degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
            if degenerate_boxes.any():
                # print the first degenerate box
                bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                degen_bb: List[float] = boxes[bb_idx].tolist()
                raise ValueError("All bounding boxes should have positive height and width."
                                 " Found invalid box {} for target at index {}."
                                 .format(degen_bb, target_idx))
        
        return targets
    
    def forward(self, features, image_size, roi_size, targets=None):
        if self.training:
            assert targets is not None
        
        # connect fpn/backbone feature to task feature
        features = self.connector(features)
        
        # Run forward on the whole slides:
        task_features, task_gts, num_anns_per_target = extract_roi_feature_maps(features, image_size, roi_size, targets=None)
        rpn_outputs = self.rpn(task_features, roi_size)
        proposals = rpn_outputs['proposals']
        task_sizes = [roi_size for _ in range(len(proposals))]
        detections, detector_losses = self.roi_heads(task_features, proposals, task_sizes, targets=None)
        detections = split_by_sizes(detections, num_anns_per_target)
        
        # Project results on the whole slides
        task_gts = split_by_sizes(task_gts, num_anns_per_target)
        outputs = [project_roi_results_on_image(r, t, image_size) for r, t in zip(detections, task_gts)]
        
        # Run forward on target rois and compute losses
        losses = {}
        if targets is not None:
            # num_anns_per_target2 = [len(_) for _ in targets]
            task_features, task_gts, num_anns_per_target = \
                extract_roi_feature_maps(features, image_size, roi_size, targets=targets)
            task_gts = self.remove_degenerate_boxes(task_gts)
            rpn_outputs = self.rpn(task_features, roi_size)
            proposal_losses = self.rpn.compute_losses(rpn_outputs, task_gts)
            
            proposals = rpn_outputs['proposals']
            task_sizes = [roi_size for _ in range(len(proposals))]
            _, detector_losses = self.roi_heads(task_features, proposals, task_sizes, task_gts)
            
            losses.update(proposal_losses)
            losses.update(detector_losses)
        
        return outputs, losses
