import os
from typing import List, Tuple, Union

import blosc2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import (
	NonDetMultiThreadedAugmenter,
)
from batchgenerators.dataloading.single_threaded_augmenter import (
	SingleThreadedAugmenter,
)
from batchgenerators.utilities.file_and_folder_operations import (
	isfile,
	join,
	load_json,
	load_pickle,
	save_json,
)
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import (
	nnUNetDatasetBlosc2,
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.crossval_split import generate_crossval_split
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.utilities.label_handling.label_handling import LabelManager
from threadpoolctl import threadpool_limits
from torch import autocast
from torch import distributed as dist

# compute canada
os.environ['nnUNet_raw'] = '/home/ranashah/scratch/MBH-SEG-2024-winning-solution/nnUNet_raw'
os.environ['nnUNet_preprocessed'] = (
	'/home/ranashah/scratch/MBH-SEG-2024-winning-solution/nnUNet_preprocessed'
)
os.environ['nnUNet_results'] = '/home/ranashah/scratch/MBH-SEG-2024-winning-solution/nnUNet_results'

csv_path = './case-wise_annotation.csv'


class nnUNetDatasetBlosc2MultiLabel(nnUNetDatasetBlosc2):
	def __init__(
		self,
		folder: str,
		csv_path: str,
		identifiers: List[str] = None,
		folder_with_segs_from_previous_stage: str = None,
	):
		super().__init__(folder, identifiers, folder_with_segs_from_previous_stage)
		self.csv_path = csv_path
		# csv header
		# patientID_studyID,any,epidural,intraparenchymal,intraventricular,subarachnoid,subdural
		# ID_00526c11_ID_d6296de728,1,0,1,0,0,0

		# ID_00526c11_ID_d6296de728 is the identifier
		# remove the column "any"

		self.csv_df = pd.read_csv(csv_path)
		self.csv_df = self.csv_df.drop(columns=['any'])
		self.csv_df['patientID_studyID'] = self.csv_df['patientID_studyID'].str.split('_').str[0]
		self.csv_df['patientID'] = self.csv_df['patientID_studyID'].str.split('_').str[0]
		self.csv_df['studyID'] = self.csv_df['patientID_studyID'].str.split('_').str[1]
		self.csv_df['patientID'] = self.csv_df['patientID'].str.split('_').str[0]

	def __getitem__(self, identifier):
		return self.load_case(identifier)

	def load_case(self, identifier):
		dparams = {'nthreads': 1}
		data_b2nd_file = join(self.source_folder, identifier + '.b2nd')

		# mmap does not work with Windows -> https://github.com/MIC-DKFZ/nnUNet/issues/2723
		mmap_kwargs = {} if os.name == 'nt' else {'mmap_mode': 'r'}
		data = blosc2.open(urlpath=data_b2nd_file, mode='r', dparams=dparams, **mmap_kwargs)

		# label for multi-label classification
		# the csv may or may not include the entry for the identifier
		# if it does, then we need to get the label from the csv
		# if it does not, then we need to return a label of all zeros
		row = self.csv_df[self.csv_df['patientID_studyID'] == identifier]
		if len(row) > 0:
			label = np.array(
				[
					row['epidural'],
					row['intraparenchymal'],
					row['intraventricular'],
					row['subarachnoid'],
					row['subdural'],
				]
			)
		else:
			label = np.zeros(5)
		properties = load_pickle(join(self.source_folder, identifier + '.pkl'))
		return data, label, properties


class nnUNetDataLoaderMultiLabel(nnUNetDataLoader):
	#
	# data: nnUNetBaseDataset,
	# batch_size: int,
	# patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
	# final_patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
	# label_manager: LabelManager,
	# oversample_foreground_percent: float = 0.0,
	# sampling_probabilities: Union[List[int], Tuple[int, ...], np.ndarray] = None,
	# pad_sides: Union[List[int], Tuple[int, ...]] = None,
	# probabilistic_oversampling: bool = False,
	# transforms=None
	def __init__(
		self,
		data: nnUNetDatasetBlosc2MultiLabel,
		batch_size: int,
		patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
		final_patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
		label_manager: LabelManager,
		oversample_foreground_percent: float = 0.0,
		sampling_probabilities: Union[List[int], Tuple[int, ...], np.ndarray] = None,
		pad_sides: Union[List[int], Tuple[int, ...]] = None,
		probabilistic_oversampling: bool = False,
		transforms=None,
	):
		"""
		DataLoader for multi-label classification.
		"""
		# call parent with dummy label_manager (not needed for multilabel)
		super().__init__(
			data=data,
			batch_size=batch_size,
			patch_size=patch_size,
			final_patch_size=final_patch_size,
			label_manager=label_manager,
			oversample_foreground_percent=oversample_foreground_percent,
			sampling_probabilities=sampling_probabilities,
			pad_sides=pad_sides,
			probabilistic_oversampling=probabilistic_oversampling,
			transforms=transforms,
		)

		# Override attributes not used in multilabel case
		self.data_shape = None  # will infer dynamically
		self.seg_shape = None
		self.has_ignore = False
		self.annotated_classes_key = None

	def determine_shapes(self):
		# For multilabel, determine data shape based on patch size
		data, label, props = self._data.load_case(self._data.identifiers[0])
		num_channels = data.shape[0]
		data_shape = (self.batch_size, num_channels, *self.patch_size)
		# labels are vectors of length 5
		label_shape = (self.batch_size, 5)
		return data_shape, label_shape

	def generate_train_batch(self):
		selected_keys = self.get_indices()
		images = []
		labels = []

		for i in selected_keys:
			data, label, properties = self._data.load_case(i)

			# crop or pad if needed
			shape = data.shape[1:]
			bbox_lbs = [0 for _ in shape]
			bbox_ubs = [min(shape[d], self.patch_size[d]) for d in range(len(shape))]
			bbox = [[l, u] for l, u in zip(bbox_lbs, bbox_ubs)]
			data_cropped = crop_and_pad_nd(data, bbox, 0)

			images.append(torch.from_numpy(data_cropped).float())
			labels.append(torch.from_numpy(label).float())

		images = torch.stack(images)
		labels = torch.stack(labels)

		if self.transforms is not None:
			with torch.no_grad():
				with threadpool_limits(limits=1, user_api=None):
					transformed_imgs = []
					for b in range(images.shape[0]):
						tmp = self.transforms(image=images[b])
						transformed_imgs.append(tmp['image'])
					images = torch.stack(transformed_imgs)

		return {'data': images, 'target': labels, 'keys': selected_keys}


class ClassificationHead(nn.Module):
	"""Classification head for multi-label classification.

	This module takes encoder features and outputs predictions for multiple binary
	classification tasks. It applies global average pooling followed by a fully
	connected layer with sigmoid activation.

	Args:
	    in_features (int): Number of input features from the encoder.
	    num_classes (int): Number of output classes for multi-label classification.
	    dropout_rate (float, optional): Dropout rate for regularization. Defaults to 0.5.

	Returns:
	    torch.Tensor: Predictions with shape (batch_size, num_classes) with sigmoid activation.

	Example:
	    >>> head = ClassificationHead(in_features=512, num_classes=5)
	    >>> encoder_features = torch.randn(8, 512, 32, 32, 16)  # (B, C, H, W, D)
	    >>> predictions = head(encoder_features)  # (8, 5)
	"""

	def __init__(self, in_features: int, num_classes: int, dropout_rate: float = 0.5):
		super().__init__()
		self.global_pool = nn.AdaptiveAvgPool3d(1)
		self.dropout = nn.Dropout(dropout_rate)
		self.classifier = nn.Linear(in_features, num_classes)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# x shape: (batch_size, channels, height, width, depth)
		x = self.global_pool(x)  # (batch_size, channels, 1, 1, 1)
		x = torch.flatten(x, 1)  # (batch_size, channels)
		x = self.dropout(x)
		x = self.classifier(x)  # (batch_size, num_classes)
		return torch.sigmoid(x)  # Apply sigmoid for multi-label classification


class nnUNetTrainerMultiLabel(nnUNetTrainer):
	"""Multi-label classification trainer based on nnUNet.

	This trainer extends the standard nnUNet trainer for multi-label classification tasks.
	It modifies the network architecture to use only the encoder with a classification head,
	changes the loss function to binary cross-entropy, and adapts the training/validation
	steps to handle multi-label targets.

	Key differences from the base trainer:
	1. Uses encoder-only architecture with classification head
	2. Uses binary cross-entropy loss for multi-label classification
	3. Uses nnUNetDatasetBlosc2MultiLabel dataset class
	4. Modifies train/validation steps for classification metrics

	Args:
	    plans (dict): nnUNet plans dictionary containing configuration.
	    configuration (str): Configuration name (e.g., '3d_fullres').
	    fold (int): Cross-validation fold number.
	    dataset_json (dict): Dataset JSON containing metadata.
	    csv_path (str): Path to CSV file containing multi-label annotations.
	    device (torch.device, optional): Training device. Defaults to CUDA.

	Example:
	    >>> trainer = nnUNetTrainerMultiLabel(
	    ...     plans=plans,
	    ...     configuration='3d_fullres',
	    ...     fold=0,
	    ...     dataset_json=dataset_json,
	    ...     csv_path='/path/to/labels.csv',
	    ... )
	    >>> trainer.initialize()
	    >>> trainer.run_training()
	"""

	def __init__(
		self,
		plans: dict,
		configuration: str,
		fold: int,
		dataset_json: dict,
		device: torch.device = torch.device('cuda'),
	):
		# Initialize parent class
		self.csv_path = csv_path
		super().__init__(plans, configuration, fold, dataset_json, device)
		self.num_classes = 5  # epidural, intraparenchymal, intraventricular, subarachnoid, subdural
		self.enable_deep_supervision = False

	def initialize(self):
		"""Initialize the trainer with custom dataset class and network architecture."""
		if not self.was_initialized:
			# Set batch size and oversampling
			self._set_batch_size_and_oversample()

			# Determine input channels
			from nnunetv2.utilities.label_handling.label_handling import (
				determine_num_input_channels,
			)

			self.num_input_channels = determine_num_input_channels(
				self.plans_manager, self.configuration_manager, self.dataset_json
			)

			# Build the segmentation network first to get encoder
			self.segmentation_network = self.build_network_architecture(
				self.configuration_manager.network_arch_class_name,
				self.configuration_manager.network_arch_init_kwargs,
				self.configuration_manager.network_arch_init_kwargs_req_import,
				self.num_input_channels,
				self.label_manager.num_segmentation_heads,
				self.enable_deep_supervision,
			)

			# Extract encoder and add classification head
			self.network = self._build_classification_network(self.segmentation_network)
			self.network = self.network.to(self.device)

			# Compile network if enabled
			if self._do_i_compile():
				self.print_to_log_file('Using torch.compile...')
				self.network = torch.compile(self.network)

			# Configure optimizers
			self.optimizer, self.lr_scheduler = self.configure_optimizers()

			# Wrap in DDP if needed
			if self.is_ddp:
				self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
				from torch.nn.parallel import DistributedDataParallel as DDP

				self.network = DDP(self.network, device_ids=[self.local_rank])

			# Build loss function
			self.loss = self._build_loss()

			# Set custom dataset class
			self.dataset_class = nnUNetDatasetBlosc2MultiLabel

			self.was_initialized = True
		else:
			raise RuntimeError('Trainer already initialized')

	def _build_classification_network(self, segmentation_network):
		"""Build classification network using encoder from segmentation network.

		Args:
		    segmentation_network: Full nnUNet segmentation network.

		Returns:
		    nn.Module: Classification network with encoder + classification head.
		"""

		# Create a wrapper that combines encoder and classification head
		class EncoderClassificationNetwork(nn.Module):
			def __init__(self, encoder, classification_head):
				super().__init__()
				self.encoder = encoder
				self.classification_head = classification_head
				# dummy decoder module
				self.decoder = nn.Module()

			def forward(self, x):
				# Get encoder features (before final segmentation layers)
				encoder_features = self.encoder(x)

				# If encoder returns multiple scales (deep supervision), use the highest resolution
				if isinstance(encoder_features, (list, tuple)):
					encoder_features = encoder_features[0]

				# Apply classification head
				return self.classification_head(encoder_features)

		# Extract encoder part - this depends on the specific architecture
		# For most nnUNet architectures, we can use the encoder directly
		if hasattr(segmentation_network, 'encoder'):
			encoder = segmentation_network.encoder
			# Get the number of features from the encoder output
			# This is architecture-dependent, we'll estimate from the decoder input
			if hasattr(segmentation_network, 'decoder') and hasattr(
				segmentation_network.decoder, 'conv_blocks_context'
			):
				# For Generic_UNet architecture
				encoder_features = segmentation_network.decoder.conv_blocks_context[
					-1
				].output_channels
			else:
				# Fallback: assume 512 features (common for nnUNet)
				encoder_features = 32
		else:
			# If no explicit encoder attribute, we'll use the whole network but modify the forward
			# This is a more general approach that should work with most architectures
			encoder = segmentation_network
			encoder_features = 32  # This might need adjustment based on actual architecture

		# Create classification head
		classification_head = ClassificationHead(
			in_features=encoder_features, num_classes=self.num_classes, dropout_rate=0.5
		)

		return EncoderClassificationNetwork(encoder, classification_head)

	def _build_loss(self):
		"""Build binary cross-entropy loss for multi-label classification.

		Returns:
		    nn.Module: Binary cross-entropy loss with logits.
		"""
		return nn.BCEWithLogitsLoss()

	def get_tr_and_val_datasets(self):
		"""Get training and validation datasets using custom multi-label dataset class.

		Returns:
		    tuple: (training_dataset, validation_dataset)
		"""
		# Create dataset split
		tr_keys, val_keys = self.do_split()

		# Load datasets with CSV path for multi-label annotations
		dataset_tr = self.dataset_class(
			folder=self.preprocessed_dataset_folder,
			csv_path=self.csv_path,
			identifiers=tr_keys,
			folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
		)
		dataset_val = self.dataset_class(
			folder=self.preprocessed_dataset_folder,
			csv_path=self.csv_path,
			identifiers=val_keys,
			folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
		)
		return dataset_tr, dataset_val

	def train_step(self, batch: dict) -> dict:
		"""Execute one training step for multi-label classification.

		Args:
		    batch (dict): Batch containing 'data' and 'target' keys.

		Returns:
		    dict: Dictionary containing loss value.
		"""
		data = batch['data']
		target = batch['target']  # This will be the multi-label target from CSV

		data = data.to(self.device, non_blocking=True)
		target = target.to(self.device, non_blocking=True).float()  # Ensure float for BCE loss

		self.optimizer.zero_grad(set_to_none=True)

		with (
			autocast(self.device.type, enabled=True)
			if self.device.type == 'cuda'
			else dummy_context()
		):
			output = self.network(data)  # Shape: (batch_size, num_classes)
			loss = self.loss(output, target)

		if self.grad_scaler is not None:
			self.grad_scaler.scale(loss).backward()
			self.grad_scaler.unscale_(self.optimizer)
			torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
			self.grad_scaler.step(self.optimizer)
			self.grad_scaler.update()
		else:
			loss.backward()
			torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
			self.optimizer.step()

		return {'loss': loss.detach().cpu().numpy()}

	def validation_step(self, batch: dict) -> dict:
		"""Execute one validation step for multi-label classification.

		Args:
		    batch (dict): Batch containing 'data' and 'target' keys.

		Returns:
		    dict: Dictionary containing loss and predictions.
		"""
		data = batch['data']
		target = batch['target']

		data = data.to(self.device, non_blocking=True)
		target = target.to(self.device, non_blocking=True).float()

		with (
			autocast(self.device.type, enabled=True)
			if self.device.type == 'cuda'
			else dummy_context()
		):
			output = self.network(data)
			loss = self.loss(output, target)

		# Convert predictions to binary (threshold at 0.5)
		predictions = (torch.sigmoid(output) > 0.5).float()

		# Calculate per-class accuracy
		correct_predictions = (predictions == target).float()
		per_class_accuracy = correct_predictions.mean(dim=0)  # Average over batch

		return {
			'loss': loss.detach().cpu().numpy(),
			'predictions': predictions.detach().cpu().numpy(),
			'targets': target.detach().cpu().numpy(),
			'per_class_accuracy': per_class_accuracy.detach().cpu().numpy(),
		}

	def on_validation_epoch_end(self, val_outputs: List[dict]):
		"""Process validation epoch results and log metrics.

		Args:
		    val_outputs (List[dict]): List of validation step outputs.
		"""
		from nnunetv2.utilities.collate_outputs import collate_outputs

		outputs = collate_outputs(val_outputs)

		if self.is_ddp:
			losses_val = [None for _ in range(dist.get_world_size())]
			dist.all_gather_object(losses_val, outputs['loss'])
			loss_here = np.vstack(losses_val).mean()

			accuracies_val = [None for _ in range(dist.get_world_size())]
			dist.all_gather_object(accuracies_val, outputs['per_class_accuracy'])
			per_class_acc = np.vstack(accuracies_val).mean(axis=0)
		else:
			loss_here = np.mean(outputs['loss'])
			per_class_acc = np.mean(outputs['per_class_accuracy'], axis=0)

		self.logger.log('val_losses', loss_here, self.current_epoch)

		# Log per-class accuracies
		class_names = [
			'epidural',
			'intraparenchymal',
			'intraventricular',
			'subarachnoid',
			'subdural',
		]
		for i, class_name in enumerate(class_names):
			self.logger.log(f'val_acc_{class_name}', per_class_acc[i], self.current_epoch)

		# Log mean accuracy across all classes
		mean_accuracy = per_class_acc.mean()
		self.logger.log('val_acc_mean', mean_accuracy, self.current_epoch)

		self.print_to_log_file(f'Validation loss: {loss_here:.4f}')
		self.print_to_log_file(f'Validation mean accuracy: {mean_accuracy:.4f}')
		for i, class_name in enumerate(class_names):
			self.print_to_log_file(f'Validation {class_name} accuracy: {per_class_acc[i]:.4f}')

	def do_split(self):
		"""
		The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
		so always the same) and save it as splits_final.json file in the preprocessed data directory.
		Sometimes you may want to create your own split for various reasons. For this you will need to create your own
		splits_final.json file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
		it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
		and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
		use a random 80:20 data split.
		:return:
		"""
		if self.dataset_class is None:
			raise ValueError('Dataset class is not set')

		if self.fold == 'all':
			# if fold==all then we use all images for training and validation
			case_identifiers = self.dataset_class.get_identifiers(self.preprocessed_dataset_folder)
			tr_keys = case_identifiers
			val_keys = tr_keys
		else:
			splits_file = join(self.preprocessed_dataset_folder_base, 'splits_final.json')
			dataset = self.dataset_class(
				self.preprocessed_dataset_folder,
				identifiers=None,
				folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
				csv_path=self.csv_path,
			)
			# if the split file does not exist we need to create it
			if not isfile(splits_file):
				self.print_to_log_file('Creating new 5-fold cross-validation split...')
				all_keys_sorted = list(np.sort(list(dataset.identifiers)))
				splits = generate_crossval_split(all_keys_sorted, seed=12345, n_splits=5)
				save_json(splits, splits_file)

			else:
				self.print_to_log_file('Using splits from existing split file:', splits_file)
				splits = load_json(splits_file)
				self.print_to_log_file(f'The split file contains {len(splits)} splits.')

			self.print_to_log_file('Desired fold for training: %d' % self.fold)
			if self.fold < len(splits):
				tr_keys = splits[self.fold]['train']
				val_keys = splits[self.fold]['val']
				self.print_to_log_file(
					'This split has %d training and %d validation cases.'
					% (len(tr_keys), len(val_keys))
				)
			else:
				self.print_to_log_file(
					'INFO: You requested fold %d for training but splits '
					'contain only %d folds. I am now creating a '
					'random (but seeded) 80:20 split!' % (self.fold, len(splits))
				)
				# if we request a fold that is not in the split file, create a random 80:20 split
				rnd = np.random.RandomState(seed=12345 + self.fold)
				keys = np.sort(list(dataset.identifiers))
				idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
				idx_val = [i for i in range(len(keys)) if i not in idx_tr]
				tr_keys = [keys[i] for i in idx_tr]
				val_keys = [keys[i] for i in idx_val]
				self.print_to_log_file(
					'This random 80:20 split has %d training and %d validation cases.'
					% (len(tr_keys), len(val_keys))
				)
			if any([i in val_keys for i in tr_keys]):
				self.print_to_log_file(
					'WARNING: Some validation cases are also in the training set. Please check the '
					'splits.json or ignore if this is intentional.'
				)
		return tr_keys, val_keys

	def get_dataloaders(self):
		if self.dataset_class is None:
			raise ValueError('Dataset class is not set')

		# we use the patch size to determine whether we need 2D or 3D dataloaders. We also use it to determine whether
		# we need to use dummy 2D augmentation (in case of 3D training) and what our initial patch size should be
		patch_size = self.configuration_manager.patch_size

		# needed for deep supervision: how much do we need to downscale the segmentation targets for the different
		# outputs?
		deep_supervision_scales = self._get_deep_supervision_scales()

		(
			rotation_for_DA,
			do_dummy_2d_data_aug,
			initial_patch_size,
			mirror_axes,
		) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

		# training pipeline
		tr_transforms = self.get_training_transforms(
			patch_size,
			rotation_for_DA,
			deep_supervision_scales,
			mirror_axes,
			do_dummy_2d_data_aug,
			use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
			is_cascaded=self.is_cascaded,
			foreground_labels=self.label_manager.foreground_labels,
			regions=self.label_manager.foreground_regions
			if self.label_manager.has_regions
			else None,
			ignore_label=self.label_manager.ignore_label,
		)

		# validation pipeline
		val_transforms = self.get_validation_transforms(
			deep_supervision_scales,
			is_cascaded=self.is_cascaded,
			foreground_labels=self.label_manager.foreground_labels,
			regions=self.label_manager.foreground_regions
			if self.label_manager.has_regions
			else None,
			ignore_label=self.label_manager.ignore_label,
		)

		dataset_tr, dataset_val = self.get_tr_and_val_datasets()

		dl_tr = nnUNetDataLoaderMultiLabel(
			dataset_tr,
			self.batch_size,
			initial_patch_size,
			self.configuration_manager.patch_size,
			self.label_manager,
			oversample_foreground_percent=self.oversample_foreground_percent,
			sampling_probabilities=None,
			pad_sides=None,
			transforms=tr_transforms,
			probabilistic_oversampling=self.probabilistic_oversampling,
		)
		dl_val = nnUNetDataLoaderMultiLabel(
			dataset_val,
			self.batch_size,
			self.configuration_manager.patch_size,
			self.configuration_manager.patch_size,
			self.label_manager,
			oversample_foreground_percent=self.oversample_foreground_percent,
			sampling_probabilities=None,
			pad_sides=None,
			transforms=val_transforms,
			probabilistic_oversampling=self.probabilistic_oversampling,
		)

		allowed_num_processes = get_allowed_n_proc_DA()
		if allowed_num_processes == 0:
			mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
			mt_gen_val = SingleThreadedAugmenter(dl_val, None)
		else:
			mt_gen_train = NonDetMultiThreadedAugmenter(
				data_loader=dl_tr,
				transform=None,
				num_processes=allowed_num_processes,
				num_cached=max(6, allowed_num_processes // 2),
				seeds=None,
				pin_memory=self.device.type == 'cuda',
				wait_time=0.002,
			)
			mt_gen_val = NonDetMultiThreadedAugmenter(
				data_loader=dl_val,
				transform=None,
				num_processes=max(1, allowed_num_processes // 2),
				num_cached=max(3, allowed_num_processes // 4),
				seeds=None,
				pin_memory=self.device.type == 'cuda',
				wait_time=0.002,
			)
		# # let's get this party started
		_ = next(mt_gen_train)
		_ = next(mt_gen_val)
		return mt_gen_train, mt_gen_val


print('nnUNetTrainerMultiLabel class created successfully!')
