use std::iter;

use pollster::FutureExt;
use wgpu::{
    CommandEncoderDescriptor, DeviceDescriptor, Features, Instance, InstanceDescriptor, Limits,
    Operations, PowerPreference, RenderPassColorAttachment, RenderPassDescriptor,
    RequestAdapterOptions, TextureViewDescriptor,
};
use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::Window,
};

fn main() {
    env_logger::init();

    let event_loop = EventLoop::new().unwrap();

    let window = Window::new(&event_loop).unwrap();

    let instance = Instance::new(InstanceDescriptor::default());

    let surface = unsafe { instance.create_surface(&window) }.unwrap();

    let adapter = instance
        .request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::None,
            force_fallback_adapter: false,
            compatible_surface: Some(&surface),
        })
        .block_on()
        .unwrap();

    let (device, queue) = adapter
        .request_device(
            &DeviceDescriptor {
                label: None,
                features: Features::default(),
                limits: Limits::downlevel_webgl2_defaults(),
            },
            None,
        )
        .block_on()
        .unwrap();

    let window_size = window.inner_size();

    let surface_configuration = surface
        .get_default_config(&adapter, window_size.width, window_size.height)
        .unwrap();

    surface.configure(&device, &surface_configuration);

    let mut minimized = false;

    event_loop
        .run(|event, elwt| match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::Resized(new_size) => {
                    if new_size.width > 0 && new_size.height > 0 {
                        let surface_configuration = surface
                            .get_default_config(&adapter, new_size.width, new_size.height)
                            .unwrap();

                        surface.configure(&device, &surface_configuration);

                        minimized = false;
                    } else {
                        minimized = true;
                    }
                }
                WindowEvent::CloseRequested => elwt.exit(),
                _ => {}
            },
            Event::AboutToWait => {
                if !minimized {
                    let surface_texture = surface.get_current_texture().unwrap();

                    let texture_view = surface_texture
                        .texture
                        .create_view(&TextureViewDescriptor::default());

                    let mut command_encoder =
                        device.create_command_encoder(&CommandEncoderDescriptor::default());

                    command_encoder.begin_render_pass(&RenderPassDescriptor {
                        color_attachments: &[Some(RenderPassColorAttachment {
                            view: &texture_view,
                            resolve_target: None,
                            ops: Operations::default(),
                        })],
                        ..Default::default()
                    });

                    let command_buffer = command_encoder.finish();

                    queue.submit(iter::once(command_buffer));

                    surface_texture.present();
                }
            }
            _ => {}
        })
        .unwrap();
}
