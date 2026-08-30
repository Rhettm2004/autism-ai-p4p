import 'package:flutter/material.dart';

import 'screens/screening_page.dart';

class AppColors {
  static const navy = Color(0xFF0B1D51);
  static const blue = Color(0xFF0B5DD7);
  static const orange = Color(0xFFF05A1A);
  static const background = Color(0xFFF5F7FB);
  static const softBlue = Color(0xFFEEF5FF);
  static const border = Color(0xFFD7DFEC);
  static const success = Color(0xFF249653);
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    final colorScheme = ColorScheme.fromSeed(
      seedColor: AppColors.blue,
      brightness: Brightness.light,
      primary: AppColors.blue,
      secondary: AppColors.orange,
      surface: Colors.white,
    );

    return MaterialApp(
      title: 'Autism AI',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: colorScheme,
        scaffoldBackgroundColor: AppColors.background,
        fontFamily: 'Arial',
        textTheme: const TextTheme(
          headlineSmall: TextStyle(
            color: AppColors.navy,
            fontWeight: FontWeight.w800,
            height: 1.15,
          ),
          titleLarge: TextStyle(
            color: AppColors.navy,
            fontWeight: FontWeight.w800,
          ),
          titleMedium: TextStyle(
            color: AppColors.navy,
            fontWeight: FontWeight.w700,
          ),
          bodyLarge: TextStyle(
            color: Color(0xFF17233F),
            fontSize: 16,
            height: 1.4,
          ),
          bodyMedium: TextStyle(
            color: Color(0xFF34405C),
            fontSize: 14,
            height: 1.4,
          ),
        ),
        cardTheme: const CardThemeData(
          color: Colors.white,
          elevation: 0,
          margin: EdgeInsets.zero,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.all(Radius.circular(18)),
            side: BorderSide(color: AppColors.border),
          ),
        ),
        inputDecorationTheme: const InputDecorationTheme(
          filled: true,
          fillColor: Colors.white,
          border: OutlineInputBorder(
            borderRadius: BorderRadius.all(Radius.circular(12)),
            borderSide: BorderSide(color: AppColors.border),
          ),
          enabledBorder: OutlineInputBorder(
            borderRadius: BorderRadius.all(Radius.circular(12)),
            borderSide: BorderSide(color: AppColors.border),
          ),
          contentPadding: EdgeInsets.symmetric(horizontal: 14, vertical: 13),
        ),
        filledButtonTheme: FilledButtonThemeData(
          style: ButtonStyle(
            minimumSize: WidgetStatePropertyAll(Size(48, 48)),
            backgroundColor: WidgetStatePropertyAll(AppColors.orange),
            foregroundColor: WidgetStatePropertyAll(Colors.white),
            textStyle: WidgetStatePropertyAll(
              TextStyle(fontWeight: FontWeight.w700),
            ),
            shape: WidgetStatePropertyAll(
              RoundedRectangleBorder(
                borderRadius: BorderRadius.all(Radius.circular(12)),
              ),
            ),
          ),
        ),
        outlinedButtonTheme: OutlinedButtonThemeData(
          style: ButtonStyle(
            minimumSize: WidgetStatePropertyAll(Size(48, 48)),
            foregroundColor: WidgetStatePropertyAll(AppColors.blue),
            side: WidgetStatePropertyAll(BorderSide(color: AppColors.blue)),
            textStyle: WidgetStatePropertyAll(
              TextStyle(fontWeight: FontWeight.w700),
            ),
            shape: WidgetStatePropertyAll(
              RoundedRectangleBorder(
                borderRadius: BorderRadius.all(Radius.circular(12)),
              ),
            ),
          ),
        ),
      ),
      home: const ScreeningPage(),
    );
  }
}
