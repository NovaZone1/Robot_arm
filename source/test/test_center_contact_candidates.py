import numpy as np

from src.perception.graspnet_runner import make_center_contact_grasp_group


def test_make_center_contact_grasp_group_uses_object_center():
    group = make_center_contact_grasp_group((0.12, -0.04, 0.51), width_m=0.06)

    assert len(group) == 1
    assert group.score == 1.0
    assert group.width == 0.06
    np.testing.assert_allclose(group.translation, (0.12, -0.04, 0.51))
    np.testing.assert_allclose(group.rotation_matrix, np.eye(3))


def test_catalog_items_prefer_center_contact_candidates():
    from src.perception.external_inference_worker import ExternalInferenceEngine

    assert ExternalInferenceEngine._use_center_contact_candidates(
        {"target_item_id": "red_block"}
    )
    assert ExternalInferenceEngine._use_center_contact_candidates(
        {"prefer_object_center_candidates": True}
    )
    assert not ExternalInferenceEngine._use_center_contact_candidates({"prompt": "cup"})
