import numpy as np
import pytest

from glassbox.data import seg_offset as S


def test_pure_helpers_offset_and_shift():
    iop = [1, 0, 0, 0, 1, 0]                                      # array x = +x, array y = +y
    ipp_ct = np.array([[0, 0, z] for z in (0.0, 2.5, 5.0)], float); ipp_seg = ipp_ct + np.array([0.0, -25.0, 0.0])
    dx, dy, dn = S.frame_offsets_mm(ipp_seg, ipp_ct, iop); assert np.allclose(dx, 0) and np.allclose(dy, -25) and np.allclose(dn, 0)
    sy, sx = S.offset_voxels(dx, dy, [0.703125, 0.703125]); assert (sy, sx) == (-36, 0)
    lab = np.zeros((2, 100, 50), np.uint8); lab[:, 60:70, 10:20] = 2; out = S.shift_label(lab, sy, sx)
    assert out.sum() == lab.sum() and (out[:, 24:34, 10:20] == 2).all() and out[:, 60:70].sum() == 0
    assert S.shift_label(lab, 0, 0).tolist() == lab.tolist()
    with pytest.raises(ValueError):
        S.offset_voxels([0, 0, 0], [0, 0, 5.0], [1, 1])            # not a rigid shift


def _synthetic_ct(n=3, rows=64, cols=64):
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    study, series, frame = generate_uid(), generate_uid(), generate_uid(); out = []
    for i in range(n):
        ds = Dataset(); ds.file_meta = FileMetaDataset(); ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"; ds.SOPInstanceUID = generate_uid(); ds.file_meta.MediaStorageSOPClassUID = ds.SOPClassUID; ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.FrameOfReferenceUID = study, series, frame; ds.Modality = "CT"; ds.Rows, ds.Columns = rows, cols; ds.PixelSpacing = [1.0, 1.0]; ds.SliceThickness = 2.5
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]; ds.ImagePositionPatient = [0.0, 0.0, 2.5 * i]; ds.InstanceNumber = i + 1; ds.SeriesNumber = 1; ds.PatientID = "P"; ds.PatientName = "P"; ds.PatientBirthDate = ""; ds.PatientSex = "O"
        ds.StudyDate = ds.StudyTime = ""; ds.AccessionNumber = ""; ds.StudyID = "1"; ds.ReferringPhysicianName = ""; ds.Manufacturer = "x"; ds.BitsAllocated = 16; ds.BitsStored = 16; ds.HighBit = 15; ds.PixelRepresentation = 1; ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"; ds.PixelData = np.zeros((rows, cols), np.int16).tobytes(); out.append(ds)
    return out


def _synthetic_seg(tmp_path, ct, offset_y_mm):
    import highdicom as hd
    from highdicom.sr.coding import CodedConcept
    px = np.zeros((len(ct), 64, 64), np.uint8); px[:, 20:30, 10:20] = 1
    desc = hd.seg.SegmentDescription(segment_number=1, segment_label="Mass", segmented_property_category=CodedConcept("85756007", "SCT", "Tissue"), segmented_property_type=CodedConcept("108369006", "SCT", "Neoplasm"),
                                     algorithm_type=hd.seg.SegmentAlgorithmTypeValues.MANUAL)
    seg = hd.seg.Segmentation(source_images=ct, pixel_array=px, segmentation_type=hd.seg.SegmentationTypeValues.BINARY, segment_descriptions=[desc], series_instance_uid=hd.UID(), series_number=2, sop_instance_uid=hd.UID(), instance_number=1,
                              manufacturer="x", manufacturer_model_name="x", software_versions="1", device_serial_number="1")
    for f in seg.PerFrameFunctionalGroupsSequence:
        f.PlanePositionSequence[0].ImagePositionPatient[1] = float(f.PlanePositionSequence[0].ImagePositionPatient[1]) + offset_y_mm
    path = str(tmp_path / "seg.dcm"); seg.save_as(path); return path, px


def test_read_uid_label_matched_positions_and_known_offset(tmp_path):
    pytest.importorskip("pydicom"); pytest.importorskip("highdicom")
    ct = _synthetic_ct(); headers = [(c, str(i)) for i, c in enumerate(ct)]
    path, px = _synthetic_seg(tmp_path, ct, 0.0); label, info = S.read_uid_label(path, headers)
    assert info["mode"] == "matched" and info["shift_voxels"] == (0, 0) and np.array_equal(label > 0, px > 0)      # matched positions: placed as drawn, no shift
    path, px = _synthetic_seg(tmp_path, ct, -5.0); label, info = S.read_uid_label(path, headers)
    assert info["mode"] == "declared_offset" and info["shift_voxels"] == (-5, 0) and abs(info["offset_mm"][1] + 5.0) < 1e-6
    assert np.array_equal(label > 0, S.shift_label(px, -5, 0) > 0) and label[:, 15:25, 10:20].all()      # the SEG pixel row 20 sits at CT row 15
