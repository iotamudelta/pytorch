#pragma once

#ifdef _WIN32
#if defined(torch_EXPORTS)
#define TORCH_API __declspec(dllexport)
#else
#define TORCH_API __declspec(dllimport)
#endif
#elif defined(__GNUC__)
#if defined(torch_EXPORTS)
#define TORCH_API __attribute__((__visibility__("default")))
#else
#define TORCH_API
#endif
#else
#define TORCH_API
#endif
