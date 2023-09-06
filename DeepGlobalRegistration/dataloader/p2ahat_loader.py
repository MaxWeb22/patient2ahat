import glob

from dataloader.base_loader import *
from dataloader.transforms import *

from util.pointcloud import get_matching_indices, make_open3d_point_cloud
from util.file import read_trajectory


class p2ahatDataset(torch.utils.data.Dataset):
    AUGMENT = None

    DATA_FILES = {
        'train': './dataloader/split/train_p2ahat.txt',
        'val': './dataloader/split/val_p2ahat.txt',
        'test': './dataloader/split/test_p2ahat.txt'
    }

    def __init__(self,
               phase,
               transform=None,
               random_rotation=True,
               random_scale=False,
               manual_seed=False,
               config=None):
        self.phase = phase
        self.files = []
        self.data_objects = []
        self.transform = transform
        self.voxel_size = 0.007
        self.matching_search_voxel_size = config.voxel_size * config.positive_pair_search_voxel_size_multiplier
        self.random_scale = random_scale
        self.min_scale = config.min_scale
        self.max_scale = config.max_scale
        self.random_rotation = random_rotation
        self.rotation_range = config.rotation_range
        self.randg = np.random.RandomState()
        if manual_seed:
            self.reset_seed()

        self.root = root = config.p2ahat_dir

        logging.info(f"Loading the subset {phase} from {root}")

        subset_names = open(self.DATA_FILES[phase]).read().split()
        for file_name in subset_names:
            source = self.root + "/source/" + str(file_name) + ".ply"
            target = self.root + "/target/" + str(file_name) + ".ply"
            tras = self.root + "/trans/" + str(file_name) + ".txt"

            self.files.append([source, target, tras])

    def __len__(self):
        return len(self.files)

    def reset_seed(self, seed=0):
        logging.info(f"Resetting the data loader seed to {seed}")
        self.randg.seed(seed)

    def __getitem__(self, idx):
        # TODO: write function

        matching_search_voxel_size = self.matching_search_voxel_size

        source = o3d.io.read_point_cloud(str(self.files[idx][0]))
        target = o3d.io.read_point_cloud(str(self.files[idx][1]))
        trans = np.loadtxt(str(self.files[idx][2]))

        xyz0 = np.array(source.points)
        xyz1 = np.array(target.points)

        xyz0_th = torch.from_numpy(xyz0)
        xyz1_th = torch.from_numpy(xyz1)

        _, sel0 = ME.utils.sparse_quantize(xyz0_th / self.voxel_size, return_index=True)
        _, sel1 = ME.utils.sparse_quantize(xyz1_th / self.voxel_size, return_index=True)

        pcd0 = make_open3d_point_cloud(torch.from_numpy(xyz0[sel0]))
        pcd1 = make_open3d_point_cloud(torch.from_numpy(xyz1[sel1]))

        matches = get_matching_indices(pcd0, pcd1, trans, matching_search_voxel_size)
        #if len(matches) < 1000:
        #    raise ValueError(f"Insufficient matches in {str(self.files[idx][0])}, {str(self.files[idx][1])}")

        #print(len(matches))

        npts0 = len(sel0)
        npts1 = len(sel1)

        feats_train0, feats_train1 = [], []

        unique_xyz0_th = xyz0_th[sel0]
        unique_xyz1_th = xyz1_th[sel1]

        feats_train0.append(torch.ones((npts0, 1)))
        feats_train1.append(torch.ones((npts1, 1)))

        feats0 = torch.cat(feats_train0, 1)
        feats1 = torch.cat(feats_train1, 1)

        coords0 = torch.floor(unique_xyz0_th / self.voxel_size)
        coords1 = torch.floor(unique_xyz1_th / self.voxel_size)

        #if self.transform:
        #    coords0, feats0 = self.transform(coords0, feats0)
        #    coords1, feats1 = self.transform(coords1, feats1)

        extra_package = {'idx': idx, 'file0': self.files[idx][0], 'file1': self.files[idx][1]}

        return (unique_xyz0_th.float(), unique_xyz1_th.float(), coords0.int(), coords1.int(), feats0.float(),
                feats1.float(), matches, trans, extra_package)
