from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainer_CE_DC_CLDC import nnUNetTrainer_CE_DC_CLDC


class nnUNetTrainer_CE_DC_CLDC_NoMirroring(nnUNetTrainer_CE_DC_CLDC):
    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        mirror_axes = None
        self.inference_allowed_mirroring_axes = None
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes
