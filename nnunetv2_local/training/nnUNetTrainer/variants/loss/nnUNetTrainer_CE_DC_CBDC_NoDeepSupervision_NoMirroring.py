from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainer_CE_DC_CBDC_NoDeepSupervision import nnUNetTrainer_CE_DC_CBDC_NoDeepSupervision

class nnUNetTrainer_CE_DC_CBDC_NoDeepSupervision_NoMirroring(nnUNetTrainer_CE_DC_CBDC_NoDeepSupervision):
    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        mirror_axes = None
        self.inference_allowed_mirroring_axes = None
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes
        