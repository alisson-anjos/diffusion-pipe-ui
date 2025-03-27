using MudBlazor.Services;
using Microsoft.EntityFrameworkCore;
using DiffusionPipeInterface.Database;
using DiffusionPipeInterface.Models;
using DiffusionPipeInterface.Utils;
using DiffusionPipeInterface.Components;
using DiffusionPipeInterface.Services;
using Microsoft.AspNetCore.Http.Features;
using DiffusionPipeInterface.ViewModels;
using System.IO.Compression;
using Microsoft.AspNetCore.Mvc;

var builder = WebApplication.CreateBuilder(args);
var connectionString = builder.Configuration.GetConnectionString("app");

// Add MudBlazor services
builder.Services.AddMudServices();


// Add services to the container.
builder.Services.AddRazorComponents(opt =>
{
    opt.DetailedErrors = true;
})
    .AddInteractiveServerComponents().AddHubOptions(opt => {
        opt.EnableDetailedErrors = true;
        opt.ClientTimeoutInterval = TimeSpan.FromMinutes(15);
        opt.MaximumReceiveMessageSize = 2000 * 1024 * 1024;
        opt.DisableImplicitFromServicesParameters = true;
    });

builder.WebHost.ConfigureKestrel(options =>
{
    options.Limits.MaxRequestBodySize = 2000 * 1024 * 1024;
});

builder.Services.Configure<FormOptions>(x =>
{
    x.ValueLengthLimit = int.MaxValue;
    x.MultipartBodyLengthLimit = int.MaxValue;
    x.MultipartBoundaryLengthLimit = int.MaxValue;
    x.MultipartHeadersCountLimit = int.MaxValue;
    x.MultipartHeadersLengthLimit = int.MaxValue;
});

builder.Services.AddDbContextFactory<AppDbContext>(options => options.UseSqlite(connectionString));
builder.Services.AddSingleton<FolderMonitorService>();
builder.Services.AddSingleton<ProcessManager>();
builder.Services.AddSingleton<InterfaceControlViewModel>();

builder.Services.AddScoped(sp => sp.GetRequiredService<IHttpClientFactory>().CreateClient("MyApp"));

builder.Services.Configure<AppSettingsConfiguration>(builder.Configuration.GetSection("Configurations"));

builder.Services.AddRazorPages(options =>
{
    options.Conventions.ConfigureFilter(new IgnoreAntiforgeryTokenAttribute());

});

builder.Services.AddAntiforgery(options => {
    options.Cookie.SameSite = SameSiteMode.None;
    options.Cookie.SecurePolicy = CookieSecurePolicy.SameAsRequest;
    options.SuppressXFrameOptionsHeader = true;
});

var appSettings = builder.Configuration.GetSection("Configurations").Get<AppSettingsConfiguration>()!;

appSettings.EnsureDirectoriesExist();

var serverUrl = appSettings.ServerUrlBase ?? "http://localhost:5000";

builder.Services.AddHttpClient("MyApp", client =>
{
    client.BaseAddress = new Uri(serverUrl);
});

var app = builder.Build();

// Configure the HTTP request pipeline.
if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Error", createScopeForErrors: true);
    // The default HSTS value is 30 days. You may want to change this for production scenarios, see https://aka.ms/aspnetcore-hsts.
    app.UseHsts();
}

app.UseHttpsRedirection();

app.Use(async (context, next) =>
{
    context.Response.Headers["Content-Security-Policy"] = "frame-ancestors 'self' *";
    context.Response.Headers["X-Frame-Options"] = "ALLOWALL";
    await next();
});

app.UseAntiforgery();

app.MapStaticAssets();
app.MapRazorComponents<App>()
    .AddInteractiveServerRenderMode();

app.MapGet("/download", (string file) =>
{
    if (System.IO.File.Exists(file))
    {
        return Results.File(file, "application/octet-stream", Path.GetFileName(file));
    }
    return Results.NotFound();
});

app.MapGet("/download-config", (string filePath) =>
{
    string configPath = Path.Combine(filePath);

    if (!System.IO.File.Exists(configPath))
    {
        return Results.NotFound("File not found.");
    }

    var filename = Path.GetFileName(filePath);

    return Results.File(configPath, "application/toml", filename);
});


app.MapPost("/prepare-download", async (HttpContext context) =>
{
    var request = await context.Request.ReadFromJsonAsync<DownloadRequest>();
    if (request == null || request.SelectedFiles == null || !request.SelectedFiles.Any())
    {
        context.Response.StatusCode = 400;
        await context.Response.WriteAsync("No files selected.");
        return;
    }

    var baseDirectory = request.BaseDirectory;
    if (string.IsNullOrEmpty(baseDirectory))
    {
        context.Response.StatusCode = 400;
        await context.Response.WriteAsync("The BaseDirectory parameter is required.");
        return;
    }

    var tempZipFile = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid()}.zip");

    using (var zipStream = new FileStream(tempZipFile, FileMode.Create))
    using (var archive = new ZipArchive(zipStream, ZipArchiveMode.Create))
    {
        foreach (var file in request.SelectedFiles)
        {
            if (File.Exists(file))
            {
                var relativePath = Path.GetRelativePath(baseDirectory, file);
                var entry = archive.CreateEntry(relativePath, CompressionLevel.Fastest);
                using var entryStream = entry.Open();
                using var fileStream = File.OpenRead(file);
                await fileStream.CopyToAsync(entryStream);
            }
        }
    }

    await context.Response.WriteAsJsonAsync(tempZipFile);
});

app.MapGet("/download-zip", async (HttpContext context) =>
{
    var zipFilePath = context.Request.Query["file"].ToString();
    if (string.IsNullOrEmpty(zipFilePath) || !File.Exists(zipFilePath))
    {
        context.Response.StatusCode = 404;
        await context.Response.WriteAsync("File not found.");
        return;
    }

    context.Response.Headers.ContentDisposition = "attachment; filename=selected_files.zip";
    context.Response.ContentType = "application/zip";

    await context.Response.SendFileAsync(zipFilePath);

    try
    {
        File.Delete(zipFilePath);
    }
    catch
    {
    }
});


app.Run();
