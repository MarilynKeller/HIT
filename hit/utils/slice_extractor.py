import os

import matplotlib
import matplotlib.cm as cm
import numpy as np
import torch
import tqdm
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image

from hit.utils.figures import tissue_palette
from hit.utils.smpl_utils import weights2colors


def add_colorbar(ax, im, fig):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    
    
def fig2img(fig, close_fig=True):
    """ Convert a Matplotlib figure to a PIL Image and return it"""
    fig.tight_layout()
    fig.savefig('/tmp/test.png')
    # load and return the image
    img = Image.open('/tmp/test.png')
    try:
        # Try a first time to get the data to avoid this bug:
        # *** OSError: unrecognized data stream contents when reading image file
        img.getdata()
    except:
        pass
    if close_fig:
        plt.close()
    return img


class SliceLevelSet():
    
    # def __init__(self, nbins=10, xbounds=[-0.2,0.2], ybounds=[-0.5,0], res=0.01):
    def __init__(self, nbins=10, xbounds=[-0.2,0.2], ybounds=[-0.2,0.2], z_bounds=[-0.2, 0.2], res=0.005):
        """ SliceLevelSet class to generate and plot slices of the level set function
        Args:
            nbins: number of bins for the contour plot
            xbounds: x axis bounds in meters
            ybounds: y axis bounds in meters
            res: resolution of the grid
        """
        
        self.nbins = nbins
        self.xbounds = xbounds
        self.ybounds = ybounds
        self.z_bounds = z_bounds
        self.res = res

    def gen_slice_points(self, z0, axis='z'):
        """ Generate an array of points on a slice at z0 (N,3)"""
        
        if axis=='y': # Frontal cut
            self.xbounds=[-0.9,0.9]
            self.ybounds=[-1.2,0.7]
            self.plane = 'frontal'
        elif axis=='z': # horizontal cut
            self.xbounds=[-0.2,0.2]
            self.ybounds=[-0.15,0.25]
            self.plane = 'horizontal'
        
        x = np.arange(self.xbounds[0], self.xbounds[1], self.res)
        y = np.arange(self.ybounds[0], self.ybounds[1], self.res)

        # Create a grid of points
        xx, yy = np.meshgrid(x, y)
        self.xx = xx
        self.yy = yy

        zz = z0 * np.ones_like(self.xx)

        if axis=='z':
            # Slice horizintally through the belly
            pts = np.stack([self.xx, zz, self.yy], axis=-1)
            
        elif axis=='y':
            #frontal slice
            pts = np.stack([self.xx, self.yy, zz], axis=-1)
        else:
            raise ValueError(f'axis must be z or y, got {axis}')
        pts_flat = pts.reshape(-1,3)  
        
        return pts_flat
    
    # def prepare_hit_values(values):
    #     """Given torch out"""

        
    def save_slice(self, im_array, name, folder):
    
        path = os.path.join(folder, f'{name}.png')
        if isinstance(im_array, torch.Tensor):
            import ipdb; ipdb.set_trace()
        elif isinstance(im_array, np.ndarray):
            im_array = np.flipud(im_array)
            Image.fromarray((im_array*255).astype(np.uint8)).save(path)
        else:
            im_array.save(path)
        print(f'Saved image to {path}')
        
        
    def process_occ(self, occupancy):
        if occupancy.shape[-1] == 4:
            occ = torch.softmax(occupancy, -1)
            occ_val = torch.argmax(occ, -1)
            occ_val = occ_val.detach().cpu().numpy()[0]
        else:
            assert isinstance(occupancy, np.ndarray)
            occ_val = occupancy
            
        return occ_val        
        
        
    def plot_disp_field(self, disp, occupancy=None, quiver=False, twod_intensities=False, is_compression=False):
        cmap = cm.rainbow 
        fig = plt.figure(figsize=(5,5))
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_aspect('equal', adjustable='box')
        ax.set_axis_off()
        fig.add_axes(ax)
        
        assert len(disp.shape) == 3 and disp.shape[-1] == 3
        
        if occupancy is not None:
            # Prepare occ
            occ_val = self.process_occ(occupancy)    
            occ_val = occ_val.reshape(self.xx.shape[0], self.xx.shape[1])
            levels=matplotlib.ticker.MaxNLocator(nbins=self.nbins).tick_values(occ_val.min(),occ_val.max())
            plt.contour(self.xx, self.yy, occ_val, colors='black', levels=levels)
        
        # Prepare disp    
        disp = disp.detach().cpu().numpy()[0]
        disp = disp.reshape(self.xx.shape[0], self.xx.shape[1], 3)
        
        c = np.linalg.norm(disp, axis=-1)
        
        if self.plane == 'frontal':
            qs = int(0.04 /self.res)
            density= 10
        else:
            qs = int(0.016 / self.res)
            density=2
          
        
            
        if self.plane == 'frontal':
            dx = disp[:,:,0]
            dy = disp[:,:,1]
            if is_compression:
                # The front visualization of the compression vector in T pose does not 
                # make sense as the vector orientation depends on the limbs pose
                dx = disp[:,:,0]*0
                dy = disp[:,:,2]*0
        else:
            dx = disp[:,:,0]
            dy = disp[:,:,2]
            if is_compression:
                # The compression vector is learned in the canonical but gives a vector in posed space
                dx = -disp[:,:,1]
                dy = -disp[:,:,0]
                            
        if quiver:
            px = self.xx[::qs, ::qs]
            py = self.yy[::qs, ::qs]
            dx = dx[::qs,::qs]
            dy = dy[::qs,::qs]
            qc = c[::qs,::qs]
            plt.quiver(px, py, dx, dy, qc, cmap=cmap, angles='xy', scale_units='xy', scale=1)
        else:
            if twod_intensities:
                # import ipdb; ipdb.set_trace()
                c = np.linalg.norm(np.stack([dx,dy], axis=2), axis=-1)
            plt.streamplot(self.xx, self.yy, dx, dy, color=c, density = density, cmap=cmap) 
            
            
        img = fig2img(fig)
        return img
    
    def plot_occupancy(self, occ):
        # Tissues occupancy
        assert len(occ.shape) ==  3 and occ.shape[-1] == 4
        # This is the 4C occupancy
        occ = torch.sigmoid(occ)
        occ = torch.argmax(occ, dim=-1)

        #color by tissue
        color_values = np.take(tissue_palette, occ[0].cpu().numpy().astype(int), axis=0)
        
        values = color_values.reshape(self.xx.shape[0], self.xx.shape[1], 4)*255
        img = Image.fromarray(values.astype(np.uint8))
        
        # Flip upside down
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
             
        return img 
    
    
    def plot_skinning_weights(self, sw, occupancy=None):
        
        fig = plt.figure()
        # skinning weights
        assert len(sw.shape) == 3 and sw.shape[-1] == 24
        sw = sw.detach().cpu().numpy()[0]
        sw = weights2colors(sw)
        sw = sw.reshape(self.xx.shape[0], self.xx.shape[1], 3)
        
        plt.imshow(sw, interpolation='bilinear', origin='lower', extent=(self.xbounds[0], self.xbounds[1], 
                                                                            self.ybounds[0], self.ybounds[1]))
        
        if occupancy is not None:
            occ_val = self.process_occ(occupancy)    
            occ_val = occ_val.reshape(self.xx.shape[0], self.xx.shape[1])
            levels=matplotlib.ticker.MaxNLocator(nbins=self.nbins).tick_values(occ_val.min(),occ_val.max())
            plt.contour(self.xx, self.yy, occ_val, colors='black', levels=levels)

        img = fig2img(fig)
        return img
    
    
    def plot_slice_value(self, values, to_plot=True, mri_values=False):
        """ Given the values array (N,3), plot the level set"""
        # cmap = cm.hsv
        cmap = cm.rainbow 
        

        # import ipdb; ipdb.set_trace()
        fig = plt.figure()
        
        if isinstance(values, torch.Tensor):
            
        
            # displacement field
            if len(values.shape) == 3 and values.shape[-1] == 3:
                values = values.detach().cpu().numpy()[0]
                values = values.reshape(self.xx.shape[0], self.xx.shape[1], 3)
                
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                fig.add_axes(ax)
                
                c = np.linalg.norm(values, axis=-1)
                
                if self.plane == 'frontal':
                    qs = 10
                else:
                    qs = 2
                plt.quiver(self.xx[::qs, ::qs], self.yy[::qs, ::qs], values[::qs,::qs,0], values[::qs,::qs,2], c[::qs,::qs],  cmap=cmap, angles='xy', scale_units='xy', scale=1)
                fig.savefig('/tmp/test.png')
                # load and return the image
                img = Image.open('/tmp/test.png')
                values = img
            
        else:

            if len(values.shape) == 1:
                values = values.reshape(self.xx.shape[0], self.xx.shape[1])
            elif len(values.shape) == 2:
                if values.shape[1] == 3:
                    values = values.reshape(self.xx.shape[0], self.xx.shape[1], 3)
                    values = np.abs(values) /  np.abs(values).max()
                if values.shape == 24:
                    values = weights2colors(values)
                    values = values.reshape(self.xx.shape[0], self.xx.shape[1], 3)
            
            plt.imshow(values, interpolation='bilinear', origin='lower', extent=(self.xbounds[0], self.xbounds[1], 
                                                                                self.ybounds[0], self.ybounds[1]), 
                    cmap=cmap)
            
            

        if mri_values:
            if to_plot:
                plt.show()
            if not type(values) == np.ndarray:
                values = values.reshape(self.xx.shape[0], self.xx.shape[1]).detach().cpu().numpy()
            else:
                values = values.reshape(self.xx.shape[0], self.xx.shape[1])
            values = np.flipud(values)
            plt.imshow(values, cmap='bone',  interpolation='nearest')
            # remove axis
            plt.axis('off')
            return values, plt.gcf()
        
        if to_plot and not mri_values:
            plt.show()
            
        plt.close()
                
        return values

    def plot_slice_levelset(self, disp, values, to_plot=True, arrow_mode='quiver', iscompression=False):
        """ Given the values array (N,3), plot the level set"""
        
        # cmap = cm.hsv
        cmap = cm.rainbow 
        
        # import ipdb; ipdb.set_trace()
        disp = disp.reshape(self.xx.shape[0], self.xx.shape[1], 3)
        
        if len(values.shape) == 1:
            values = values.reshape(self.xx.shape[0], self.xx.shape[1])
        else:
            values = values.reshape(self.xx.shape[0], self.xx.shape[1], 3)
            values = np.abs(values) / 0.05

        # plt.figure()
        # plt.imshow(values, interpolation='bilinear', origin='lower', extent=(self.bounds[0], self.bounds[1], self.bounds[0], self.bounds[1]), cmap=cm.bwr)
    
        # # plt.quiver(self.xx, self.yy, disp[:,:,0], disp[:,:,1], color='r')
        # # plt.streamplot(self.xx, self.yy, disp[:,:,0], disp[:,:,1], color='r')
        # plt.streamplot(self.xx, self.yy, disp[:,:,0], disp[:,:,1], color=values, density = 2, cmap=cm.hsv)
        # plt.colorbar()

        # # plt.contourf(x,y,w,levels=levels)
        # plt.contour(self.xx, self.yy, values, colors='black', levels=levels)
        
        # Create two subplots. One with the imshow, one with streamplot
        fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(20,7))
        
        if len(values.shape) == 2:
            im = ax0.imshow(values, interpolation='bilinear', origin='lower', extent=(self.xbounds[0], self.xbounds[1], self.ybounds[0], self.ybounds[1]), cmap=cmap)
            add_colorbar(ax0, im, fig)
            levels=matplotlib.ticker.MaxNLocator(nbins=self.nbins).tick_values(values.min(),values.max())
            ax0.contour(self.xx, self.yy, values, colors='black', levels=levels)
        else:
            # v2 = values[:,:,1] = 0
            im = ax0.imshow(values, interpolation='bilinear', origin='lower', extent=(self.xbounds[0], self.xbounds[1], self.ybounds[0], self.ybounds[1]))

        # import ipdb; ipdb.set_trace()
        ax0.set_title('imshow')
        
        c = np.linalg.norm(disp, axis=-1)
        # if arrow_mode == 'quiver':
        
        if iscompression:
            disp_flipped = np.zeros_like(disp)
            disp_flipped[:,:,0] = disp[:,:,2]
            disp_flipped[:,:,2] = -disp[:,:,0]
            disp=disp_flipped
        
        # ax1.quiver(self.xx, self.yy, disp[:,:,0], disp[:,:,2], c,  cmap=cmap, angles='xy', scale_units='xy', scale=1)
        # Only plot half the arrows
        ax1.quiver(self.xx[::2, ::2], self.yy[::2, ::2], disp[::2,::2,0], disp[::2,::2,2], c[::2,::2],  cmap=cmap, angles='xy', scale_units='xy', scale=1)
        print(disp[:,:,0].max(), disp[:,:,1].max(), disp[:,:,2].max())
        ax1.set_title('quiver')
        # elif arrow_mode == 'streamplot':
        ax2.streamplot(self.xx, self.yy, disp[:,:,0], disp[:,:,2], color=c, density = 2, cmap=cmap) #, linewidth=lw Input is x,z,y so I should take the disp along x and y
        ax2.set_title('streamplot')
        # else:
        if len(values.shape) == 2:
            for ax in [ax1, ax2]:
                #     raise ValueError(f'arrow_mode not recognized, must be quiver or streamplot, got {arrow_mode}')
                # add_colorbar(ax, im, fig)
                ax.contour(self.xx, self.yy, values, colors='black', levels=levels)
                ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        
        
        # plt.grid('on')
        if to_plot:
            plt.show()


    def eval_slice(self, z0, f):
        
        pts_flat = self.gen_slice_points(z0) 
        disp = f(pts_flat) 
        
        values = np.linalg.norm(disp[:,[0,2]], axis=-1)  
        self.plot_slice_levelset(disp, values)
        
        self.plot_slice_value(values, to_plot=True)
        
    
    
if __name__ == '__main__':
    
    z0 = 0
    f = lambda pts: np.sin(pts[:,0])**10+np.cos(10+pts[:,1]*pts[:,1])*np.cos(pts[:,0])
    g = lambda pts: pts
    h = lambda pts: np.dstack([np.sin(pts[:,0]),  pts[:,1], np.sin(pts[:,2])])[0]
    
    sl = SliceLevelSet()

    
    sl.eval_slice(z0, h) # 3 channel output
    # save plt figure
    plt.savefig('/tmp/slice_levelset.png')
    print('saved /tmp/slice_levelset.png')
    
