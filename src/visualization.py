import vtk
import numpy as np
from pathlib import Path
import itk

class VTKObjectFactory:
    """Factory class for creating VTK objects with standard configurations."""
    
    def create_sphere(self, center, radius):
        """Create a sphere source."""
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(center)
        sphere.SetRadius(radius)
        sphere.SetPhiResolution(20)
        sphere.SetThetaResolution(20)
        sphere.Update()
        return sphere.GetOutput()
    
    def create_actor(self, poly_data, color, opacity=1.0):
        """Create an actor from polydata."""
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly_data)
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color)
        actor.GetProperty().SetOpacity(opacity)
        return actor
    
    def create_volume_actor(self, image_data, color_function, opacity_function):
        """Create a volume actor with transfer functions."""
        volume_property = vtk.vtkVolumeProperty()
        volume_property.SetColor(color_function)
        volume_property.SetScalarOpacity(opacity_function)
        volume_property.SetInterpolationTypeToLinear()
        volume_property.ShadeOn()
        volume_property.SetAmbient(0.4)
        volume_property.SetDiffuse(0.6)
        volume_property.SetSpecular(0.2)
        
        mapper = vtk.vtkGPUVolumeRayCastMapper()
        mapper.SetInputData(image_data)
        
        volume = vtk.vtkVolume()
        volume.SetMapper(mapper)
        volume.SetProperty(volume_property)
        
        return volume
    
    def create_surface_actor(self, image_data, iso_value=127, color=(1, 0, 0), opacity=0.8):
        """Create a surface actor using marching cubes."""
        marching_cubes = vtk.vtkMarchingCubes()
        marching_cubes.SetInputData(image_data)
        marching_cubes.SetValue(0, iso_value)
        marching_cubes.Update()
        
        smoother = vtk.vtkWindowedSincPolyDataFilter()
        smoother.SetInputConnection(marching_cubes.GetOutputPort())
        smoother.SetNumberOfIterations(15)
        smoother.BoundarySmoothingOff()
        smoother.FeatureEdgeSmoothingOff()
        smoother.SetFeatureAngle(120.0)
        smoother.SetPassBand(0.001)
        smoother.NonManifoldSmoothingOn()
        smoother.NormalizeCoordinatesOn()
        smoother.Update()
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(smoother.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color)
        actor.GetProperty().SetOpacity(opacity)
        
        return actor


class ITKToVTKConverter:
    """Converter class for ITK to VTK image data."""
    
    @staticmethod
    def convert_itk_to_vtk(itk_image):
        try:
            region = itk_image.GetLargestPossibleRegion()
            size = region.GetSize()
            origin = itk_image.GetOrigin()
            spacing = itk_image.GetSpacing()
            
            np_array = itk.array_from_image(itk_image)
            
            vtk_image = vtk.vtkImageData()
            vtk_image.SetDimensions(size[0], size[1], size[2])
            vtk_image.SetOrigin(origin[0], origin[1], origin[2])
            vtk_image.SetSpacing(spacing[0], spacing[1], spacing[2])
            
            vtk_array = vtk.vtkFloatArray()
            vtk_array.SetName("ImageData")
            vtk_array.SetNumberOfTuples(np_array.size)
            
            for i in range(np_array.size):
                vtk_array.SetValue(i, np_array.flat[i])
            
            vtk_image.GetPointData().SetScalars(vtk_array)
            
            return vtk_image
            
        except Exception as e:
            print(f"Error converting ITK to VTK: {e}")
            return None


class TumorVisualization:
    """Main visualization class for tumor analysis."""
    
    def __init__(self):
        self.factory = VTKObjectFactory()
        self.converter = ITKToVTKConverter()
        
        self.renderer = vtk.vtkRenderer()
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetSize(1200, 800)
        self.render_window.SetWindowName("Longitudinal Tumor Analysis")
        
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)
        
        self.setup_scene()
    
    def setup_scene(self):
        """Set up the basic scene with lighting and camera."""
        
        self.renderer.SetBackground(0.1, 0.1, 0.2)
        
        self.renderer.SetLayer(0)
        self.render_window.SetNumberOfLayers(2)
        
        light = vtk.vtkLight()
        light.SetPosition(1, 1, 1)
        light.SetFocalPoint(0, 0, 0)
        light.SetColor(1, 1, 1)
        light.SetIntensity(1.0)
        self.renderer.AddLight(light)
        
        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(0, 0, 300)
        camera.SetFocalPoint(0, 0, 0)
        camera.SetViewUp(0, 1, 0)
    
    def visualize_tumors(self, brain_file1, brain_file2, tumor_file1, tumor_file2, analysis_results):
        """Main visualization method."""
        print("=== Tumor Visualization ===")
        
        try:
            brain1 = self.load_image(brain_file1)
            brain2 = self.load_image(brain_file2)
            tumor1 = self.load_image(tumor_file1)
            tumor2 = self.load_image(tumor_file2)
            
            if not all([brain1, brain2, tumor1, tumor2]):
                print("Error loading images, using simplified visualization")
                self.create_simplified_visualization(analysis_results)
                return
            
            vtk_brain1 = self.converter.convert_itk_to_vtk(brain1)
            vtk_brain2 = self.converter.convert_itk_to_vtk(brain2)
            vtk_tumor1 = self.converter.convert_itk_to_vtk(tumor1)
            vtk_tumor2 = self.converter.convert_itk_to_vtk(tumor2)
            
            self.create_brain_visualization(vtk_brain1, vtk_brain2)
            self.create_tumor_visualization(vtk_tumor1, vtk_tumor2)
            self.add_annotations(analysis_results)
            
            self.renderer.ResetCamera()
            self.render_window.Render()
            self.interactor.Start()
            
        except Exception as e:
            print(f"Error in main visualization: {e}")
            self.create_simplified_visualization(analysis_results)
    
    def load_image(self, file_path):
        """Load image using ITK."""
        try:
            if not Path(file_path).exists():
                print(f"File not found: {file_path}")
                return None
            
            if file_path.endswith('.nrrd'):
                ImageType = itk.Image[itk.F, 3]
                reader = itk.ImageFileReader[ImageType].New()
            else:
                print(f"Unsupported file format: {file_path}")
                return None
            
            reader.SetFileName(file_path)
            reader.Update()
            return reader.GetOutput()
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def create_brain_visualization(self, vtk_brain1, vtk_brain2):
        """Create brain volume visualization with distinct colors for each scan."""
        try:
            brain1_range = vtk_brain1.GetScalarRange()
            brain2_range = vtk_brain2.GetScalarRange()
            
            color_tf1 = vtk.vtkColorTransferFunction()
            color_tf1.AddRGBPoint(brain1_range[0], 0.0, 0.0, 0.0)
            color_tf1.AddRGBPoint(brain1_range[1] * 0.2, 0.4, 0.2, 0.2)  
            color_tf1.AddRGBPoint(brain1_range[1] * 0.5, 0.6, 0.4, 0.4)  
            color_tf1.AddRGBPoint(brain1_range[1] * 0.8, 0.8, 0.6, 0.6)  
            color_tf1.AddRGBPoint(brain1_range[1], 1.0, 0.8, 0.8)        
            
            opacity_tf1 = vtk.vtkPiecewiseFunction()
            opacity_tf1.AddPoint(brain1_range[0], 0.0)
            opacity_tf1.AddPoint(brain1_range[1] * 0.3, 0.01)
            opacity_tf1.AddPoint(brain1_range[1] * 0.7, 0.08)  
            opacity_tf1.AddPoint(brain1_range[1], 0.2)
            
            brain_volume1 = self.factory.create_volume_actor(vtk_brain1, color_tf1, opacity_tf1)
            brain_volume1.SetPosition(-100, 0, 0)
            self.renderer.AddVolume(brain_volume1)
            
            color_tf2 = vtk.vtkColorTransferFunction()
            color_tf2.AddRGBPoint(brain2_range[0], 0.0, 0.0, 0.0)
            color_tf2.AddRGBPoint(brain2_range[1] * 0.2, 0.2, 0.4, 0.2)  
            color_tf2.AddRGBPoint(brain2_range[1] * 0.5, 0.4, 0.6, 0.4)  
            color_tf2.AddRGBPoint(brain2_range[1] * 0.8, 0.6, 0.8, 0.6)  
            color_tf2.AddRGBPoint(brain2_range[1], 0.8, 1.0, 0.8)        
            
            opacity_tf2 = vtk.vtkPiecewiseFunction()
            opacity_tf2.AddPoint(brain2_range[0], 0.0)
            opacity_tf2.AddPoint(brain2_range[1] * 0.3, 0.01)
            opacity_tf2.AddPoint(brain2_range[1] * 0.7, 0.08)  
            opacity_tf2.AddPoint(brain2_range[1], 0.2)
            
            brain_volume2 = self.factory.create_volume_actor(vtk_brain2, color_tf2, opacity_tf2)
            brain_volume2.SetPosition(100, 0, 0)
            self.renderer.AddVolume(brain_volume2)
            
            print("Brain volumes added successfully with distinct colors")
            
        except Exception as e:
            print(f"Error in brain visualization: {e}")
    
    def create_tumor_visualization(self, vtk_tumor1, vtk_tumor2):
        """Create tumor surface visualization with enhanced visibility."""
        try:
            tumor_actor1 = self.factory.create_surface_actor(
                vtk_tumor1, 
                iso_value=127, 
                color=(1.0, 0.0, 0.0),  
                opacity=0.9  
            )
            tumor_actor1.SetPosition(-100, 0, 0)
            
            tumor_actor1.GetProperty().SetSpecular(0.3)
            tumor_actor1.GetProperty().SetSpecularPower(20)
            self.renderer.AddActor(tumor_actor1)
            
            tumor_actor2 = self.factory.create_surface_actor(
                vtk_tumor2, 
                iso_value=127, 
                color=(0.0, 1.0, 0.0),  
                opacity=0.9  
            )
            tumor_actor2.SetPosition(100, 0, 0)
            
            tumor_actor2.GetProperty().SetSpecular(0.3)
            tumor_actor2.GetProperty().SetSpecularPower(20)
            self.renderer.AddActor(tumor_actor2)
            
            print("Tumor surfaces added successfully")
            
        except Exception as e:
            print(f"Error in tumor visualization: {e}")
    
    def add_annotations(self, analysis_results):
        """Add comprehensive text annotations and bar chart to the visualization."""
        try:
            dice_coeff = analysis_results.get('dice_coefficient', 0)
            jaccard_index = analysis_results.get('jaccard_index', 0)
            volume1 = analysis_results.get('volume1', 0)
            volume2 = analysis_results.get('volume2', 0)
            
            if isinstance(dice_coeff, (int, float)):
                dice_str = f"{dice_coeff:.3f}"
            else:
                dice_str = str(dice_coeff)
            
            if isinstance(jaccard_index, (int, float)):
                jaccard_str = f"{jaccard_index:.3f}"
            else:
                jaccard_str = str(jaccard_index)
            
            if isinstance(volume1, (int, float)):
                vol1_str = f"{volume1:.0f} mm³"
            else:
                vol1_str = str(volume1)
            
            if isinstance(volume2, (int, float)):
                vol2_str = f"{volume2:.0f} mm³"
            else:
                vol2_str = str(volume2)
            
            if isinstance(volume1, (int, float)) and isinstance(volume2, (int, float)):
                vol_diff = volume2 - volume1
                vol_diff_str = f"{vol_diff:+.0f} mm³"
                vol_percent = (vol_diff / volume1) * 100 if volume1 > 0 else 0
                vol_percent_str = f"{vol_percent:+.1f}%"
            else:
                vol_diff_str = "N/A"
                vol_percent_str = "N/A"
            
            results_actor = vtk.vtkTextActor()
            results_content = "TUMOR ANALYSIS RESULTS\n"
            results_content += "=" * 30 + "\n\n"
            results_content += f"METRICS:\n"
            results_content += f"  Dice Coefficient: {dice_str}\n"
            results_content += f"  Jaccard Index: {jaccard_str}\n\n"
            results_content += f"VOLUMES:\n"
            results_content += f"  Scan 1: {vol1_str}\n"
            results_content += f"  Scan 2: {vol2_str}\n"
            results_content += f"  Change: {vol_diff_str} ({vol_percent_str})\n"
            
            results_actor.SetInput(results_content)
            results_actor.SetPosition(10, 400)
            results_actor.GetTextProperty().SetFontSize(12)
            results_actor.GetTextProperty().SetColor(1, 1, 1)
            results_actor.GetTextProperty().SetFontFamilyToArial()
            
            self.renderer.AddViewProp(results_actor)
            
            scan1_label = vtk.vtkTextActor()
            scan1_label.SetInput("SCAN 1 (BASELINE)\nRed Tumor\nLeft Side")
            scan1_label.SetPosition(10, 120)
            scan1_label.GetTextProperty().SetFontSize(14)
            scan1_label.GetTextProperty().SetColor(1, 0.2, 0.2)
            scan1_label.GetTextProperty().SetFontFamilyToArial()
            scan1_label.GetTextProperty().SetBold(1)
            
            scan2_label = vtk.vtkTextActor()
            scan2_label.SetInput("SCAN 2 (FOLLOW-UP)\nGreen Tumor\nRight Side")
            scan2_label.SetPosition(600, 120)
            scan2_label.GetTextProperty().SetFontSize(14)
            scan2_label.GetTextProperty().SetColor(0.2, 1, 0.2)
            scan2_label.GetTextProperty().SetFontFamilyToArial()
            scan2_label.GetTextProperty().SetBold(1)
            
            self.renderer.AddViewProp(scan1_label)
            self.renderer.AddViewProp(scan2_label)
            
            title_actor = vtk.vtkTextActor()
            title_actor.SetInput("LONGITUDINAL BRAIN TUMOR ANALYSIS")
            title_actor.SetPosition(600, 750)
            title_actor.GetTextProperty().SetFontSize(20)
            title_actor.GetTextProperty().SetColor(1, 1, 0)
            title_actor.GetTextProperty().SetFontFamilyToArial()
            title_actor.GetTextProperty().SetBold(1)
            title_actor.GetTextProperty().SetJustificationToCentered()
            
            self.renderer.AddViewProp(title_actor)
            
            method_actor = vtk.vtkTextActor()
            method_actor.SetInput("ITK Registration + Percentile Segmentation + VTK Visualization")
            method_actor.SetPosition(600, 30)
            method_actor.GetTextProperty().SetFontSize(10)
            method_actor.GetTextProperty().SetColor(0.8, 0.8, 0.8)
            method_actor.GetTextProperty().SetFontFamilyToArial()
            method_actor.GetTextProperty().SetJustificationToCentered()
            
            self.renderer.AddViewProp(method_actor)
            
            self._add_volume_bar_chart(volume1, volume2)
            
            print("Annotations added successfully")
            
        except Exception as e:
            print(f"Error adding annotations: {e}")
            
            self._add_basic_annotations(analysis_results)
    
    def _add_volume_bar_chart(self, volume1, volume2):
        """Add a fixed bar chart showing volume comparison."""
        try:
            if not isinstance(volume1, (int, float)) or not isinstance(volume2, (int, float)):
                return
            
            chart_x = 850    
            chart_y = 100    
            chart_width = 120
            chart_height = 150
            
            max_volume = max(volume1, volume2) if max(volume1, volume2) > 0 else 1
            
            bar1_height = (volume1 / max_volume) * chart_height
            bar2_height = (volume2 / max_volume) * chart_height
            
            bar1_points = vtk.vtkPoints()
            bar1_points.InsertNextPoint(chart_x, chart_y, 0)
            bar1_points.InsertNextPoint(chart_x + 40, chart_y, 0)
            bar1_points.InsertNextPoint(chart_x + 40, chart_y + bar1_height, 0)
            bar1_points.InsertNextPoint(chart_x, chart_y + bar1_height, 0)
            
            bar1_cells = vtk.vtkCellArray()
            bar1_cells.InsertNextCell(4)
            bar1_cells.InsertCellPoint(0)
            bar1_cells.InsertCellPoint(1)
            bar1_cells.InsertCellPoint(2)
            bar1_cells.InsertCellPoint(3)
            
            bar1_poly = vtk.vtkPolyData()
            bar1_poly.SetPoints(bar1_points)
            bar1_poly.SetPolys(bar1_cells)
            
            bar1_mapper = vtk.vtkPolyDataMapper2D()
            bar1_mapper.SetInputData(bar1_poly)
            
            bar1_actor = vtk.vtkActor2D()
            bar1_actor.SetMapper(bar1_mapper)
            bar1_actor.GetProperty().SetColor(1.0, 0.2, 0.2)
            bar1_actor.GetPositionCoordinate().SetCoordinateSystemToDisplay()
            
            self.renderer.AddActor(bar1_actor)
            
            bar2_points = vtk.vtkPoints()
            bar2_points.InsertNextPoint(chart_x + 60, chart_y, 0)
            bar2_points.InsertNextPoint(chart_x + 100, chart_y, 0)
            bar2_points.InsertNextPoint(chart_x + 100, chart_y + bar2_height, 0)
            bar2_points.InsertNextPoint(chart_x + 60, chart_y + bar2_height, 0)
            
            bar2_cells = vtk.vtkCellArray()
            bar2_cells.InsertNextCell(4)
            bar2_cells.InsertCellPoint(0)
            bar2_cells.InsertCellPoint(1)
            bar2_cells.InsertCellPoint(2)
            bar2_cells.InsertCellPoint(3)
            
            bar2_poly = vtk.vtkPolyData()
            bar2_poly.SetPoints(bar2_points)
            bar2_poly.SetPolys(bar2_cells)
            
            bar2_mapper = vtk.vtkPolyDataMapper2D()
            bar2_mapper.SetInputData(bar2_poly)
            
            bar2_actor = vtk.vtkActor2D()
            bar2_actor.SetMapper(bar2_mapper)
            bar2_actor.GetProperty().SetColor(0.2, 1.0, 0.2)
            bar2_actor.GetPositionCoordinate().SetCoordinateSystemToDisplay()
            
            self.renderer.AddActor(bar2_actor)
            
            chart_title = vtk.vtkTextActor()
            chart_title.SetInput("Volume Comparison (mm³)")
            chart_title.SetPosition(chart_x + 50, chart_y + chart_height + 20)
            chart_title.GetTextProperty().SetFontSize(14)
            chart_title.GetTextProperty().SetColor(1, 1, 1)
            chart_title.GetTextProperty().SetFontFamilyToArial()
            chart_title.GetTextProperty().SetBold(1)
            chart_title.GetTextProperty().SetJustificationToCentered()
            
            self.renderer.AddActor(chart_title)
            
            val1_label = vtk.vtkTextActor()
            val1_label.SetInput(f"{volume1:.0f}")
            val1_label.SetPosition(chart_x + 20, chart_y + bar1_height + 5)
            val1_label.GetTextProperty().SetFontSize(10)
            val1_label.GetTextProperty().SetColor(1, 0.2, 0.2)
            val1_label.GetTextProperty().SetJustificationToCentered()
            
            val2_label = vtk.vtkTextActor()
            val2_label.SetInput(f"{volume2:.0f}")
            val2_label.SetPosition(chart_x + 80, chart_y + bar2_height + 5)
            val2_label.GetTextProperty().SetFontSize(10)
            val2_label.GetTextProperty().SetColor(0.2, 1, 0.2)
            val2_label.GetTextProperty().SetJustificationToCentered()
            
            self.renderer.AddActor(val1_label)
            self.renderer.AddActor(val2_label)
            
            scan1_axis = vtk.vtkTextActor()
            scan1_axis.SetInput("Scan 1")
            scan1_axis.SetPosition(chart_x + 20, chart_y - 20)
            scan1_axis.GetTextProperty().SetFontSize(10)
            scan1_axis.GetTextProperty().SetColor(1, 1, 1)
            scan1_axis.GetTextProperty().SetJustificationToCentered()
            
            scan2_axis = vtk.vtkTextActor()
            scan2_axis.SetInput("Scan 2")
            scan2_axis.SetPosition(chart_x + 80, chart_y - 20)
            scan2_axis.GetTextProperty().SetFontSize(10)
            scan2_axis.GetTextProperty().SetColor(1, 1, 1)
            scan2_axis.GetTextProperty().SetJustificationToCentered()
            
            self.renderer.AddActor(scan1_axis)
            self.renderer.AddActor(scan2_axis)
            
            print("Volume bar chart added successfully with fixed positioning")
            
        except Exception as e:
            print(f"Error adding bar chart: {e}")
    
    def _add_basic_annotations(self, analysis_results):
        """Fallback method for basic annotations."""
        try:
            text_actor = vtk.vtkTextActor()
            text_content = f"Basic Analysis Results:\n"
            text_content += f"Volume 1: {analysis_results.get('volume1', 'N/A')}\n"
            text_content += f"Volume 2: {analysis_results.get('volume2', 'N/A')}\n"
            text_content += f"Change: {analysis_results.get('volume_change', 'N/A')}"
            
            text_actor.SetInput(text_content)
            text_actor.SetPosition(10, 10)
            text_actor.GetTextProperty().SetFontSize(12)
            text_actor.GetTextProperty().SetColor(1, 1, 1)
            
            self.renderer.AddViewProp(text_actor)
            
        except Exception as e:
            print(f"Error in fallback annotations: {e}")
    
    def create_simplified_visualization(self, analysis_results):
        """Create simplified visualization using geometric shapes."""
        try:
            print("Creating simplified tumor visualization...")
            
            vol1 = analysis_results.get('volume1', 50000)
            vol2 = analysis_results.get('volume2', 55000)
            
            radius1 = ((3 * vol1) / (4 * np.pi)) ** (1/3) * 0.01  
            radius2 = ((3 * vol2) / (4 * np.pi)) ** (1/3) * 0.01
            
            sphere1 = self.factory.create_sphere((-50, 0, 0), radius1)
            actor1 = self.factory.create_actor(sphere1, (1, 0.2, 0.2))
            
            sphere2 = self.factory.create_sphere((50, 0, 0), radius2)
            actor2 = self.factory.create_actor(sphere2, (0.2, 1, 0.2))
            
            self.renderer.AddActor(actor1)
            self.renderer.AddActor(actor2)
            
            self.add_annotations(analysis_results)
            
            self.renderer.ResetCamera()
            self.render_window.Render()
            self.interactor.Start()
            
            print("Simplified visualization completed successfully")
            
        except Exception as e:
            print(f"Error in simplified visualization: {e}")

def visualize_tumor_analysis(brain_file1, brain_file2, tumor_file1, tumor_file2, analysis_results):
    """Main function to create tumor visualization."""
    try:
        visualizer = TumorVisualization()
        visualizer.visualize_tumors(brain_file1, brain_file2, tumor_file1, tumor_file2, analysis_results)
    except Exception as e:
        print(f"Visualization error: {e}")
        print("Please check VTK installation and image files")

if __name__ == "__main__":
    
    sample_results = {
        'volume1': 50000,
        'volume2': 55000,
        'volume_change': '+10%',
        'dice_coefficient': 0.85
    }
    
    visualizer = TumorVisualization()
    visualizer.create_simplified_visualization(sample_results)
