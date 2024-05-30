load("@mediapipe//mediapipe:platforms.bzl", "config_setting_and_platform")

licenses(["notice"])

# Linux 64-bit.
config_setting_and_platform(
    name = "linux",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
    ],
    visibility = ["//visibility:public"],
)

exports_files(
    ["provisioning_profile.mobileprovision"],
    visibility = ["//visibility:public"],
)
