using Microsoft.AspNetCore.Authentication.Cookies;
using Microsoft.AspNetCore.Identity;
using Microsoft.EntityFrameworkCore;
using Tiamat.Core.Services.Interfaces;
using Tiamat.Core.Services;
using Tiamat.DataAccess;
using Tiamat.Models;
using Tiamat.Core;

var builder = WebApplication.CreateBuilder(args);

var connectionString = builder.Configuration.GetConnectionString("DefaultConnection")
    ?? throw new InvalidOperationException("Connection string 'DefaultConnection' not found.");

builder.Services.AddDbContext<TiamatDbContext>(options =>
    options.UseSqlServer(connectionString, sqlOptions =>
    {
        sqlOptions.MigrationsAssembly("Tiamat.DataAccess");
        sqlOptions.EnableRetryOnFailure(
            maxRetryCount: 5,
            maxRetryDelay: TimeSpan.FromSeconds(30),
            errorNumbersToAdd: null);
    })
);
builder.Services.AddDatabaseDeveloperPageExceptionFilter();
builder.Services.AddIdentity<User, IdentityRole<Guid>>(options =>
{
    options.Password.RequiredLength = 8;
    options.Password.RequireDigit = true;
    options.Password.RequireUppercase = true;
    options.Password.RequireLowercase = true;
    options.Password.RequireNonAlphanumeric = true;
    options.Password.RequiredUniqueChars = 1;
    options.Lockout.DefaultLockoutTimeSpan = TimeSpan.FromMinutes(5);
    options.Lockout.MaxFailedAccessAttempts = 5;
    options.Lockout.AllowedForNewUsers = true;
    options.User.RequireUniqueEmail = true;
    options.SignIn.RequireConfirmedAccount = false;
})
.AddEntityFrameworkStores<TiamatDbContext>()
.AddDefaultTokenProviders();
builder.Services.AddAuthorization();
builder.Services.AddControllersWithViews();
builder.Services.AddRazorPages();
builder.Services.AddHostedService<SeedDatabase>();

builder.Services.AddHttpClient();
builder.Services.AddSingleton<IPythonApiService, PythonApiService>();
builder.Services.AddScoped<CheckPythonConnectionAttribute>();

builder.Services.AddScoped<IAccountService>(provider => {
    var dbContext = provider.GetRequiredService<TiamatDbContext>();
    var notificationService = provider.GetRequiredService<INotificationService>();
    var pythonApiService = provider.GetRequiredService<IPythonApiService>();
    var logger = provider.GetRequiredService<ILogger<AccountService>>();
    return new AccountService(dbContext, notificationService, pythonApiService, logger);
});
builder.Services.AddScoped<IPositionService, PositionService>();
builder.Services.AddScoped<IAccountSettingService, AccountSettingService>();
builder.Services.AddScoped<INotificationService, NotificationService>();
builder.Services.AddHostedService<PythonInitializationService>();

builder.Services.AddControllers()
    .AddJsonOptions(options =>
    {
        options.JsonSerializerOptions.PropertyNamingPolicy = System.Text.Json.JsonNamingPolicy.CamelCase;
    });

builder.Services.AddAuthentication(CookieAuthenticationDefaults.AuthenticationScheme)
    .AddCookie(options =>
    {
        options.LoginPath = "/User/Login";
    });

var app = builder.Build();

if (app.Environment.IsDevelopment())
{
    app.UseMigrationsEndPoint();
}
else
{
    app.UseExceptionHandler("/Home/Error");
    app.UseHsts();
}

app.UseHttpsRedirection();
app.UseStaticFiles();

app.UseRouting();
app.UseAuthentication();
app.UseAuthorization();

app.MapControllerRoute(
    name: "default",
    pattern: "{controller=Home}/{action=Login}/{id?}");
app.MapRazorPages();
app.MapControllers();

app.Run();