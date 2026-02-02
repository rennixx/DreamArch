import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { style } = body;

    if (!style) {
      return NextResponse.json(
        { detail: 'Style parameter is required' },
        { status: 400 }
      );
    }

    // Create form data for backend
    const formData = new FormData();
    formData.append('style', style);

    // Proxy to backend
    const backendUrl = 'http://localhost:8000/api/beats/generate';

    const response = await fetch(backendUrl, {
      method: 'POST',
      body: formData,
      // Don't set Content-Type header, let browser set it with boundary
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error('Backend error:', response.status, errorText);

      return NextResponse.json(
        { detail: errorText || 'Backend request failed' },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);

  } catch (error) {
    console.error('API route error:', error);
    return NextResponse.json(
      { detail: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    );
  }
}
