package org.tensorflow.lite.examples.poseestimation

import org.tensorflow.lite.examples.poseestimation.org.tensorflow.lite.examples.poseestimation.LoginRequest
import org.tensorflow.lite.examples.poseestimation.org.tensorflow.lite.examples.poseestimation.LoginResponse
import org.tensorflow.lite.examples.poseestimation.org.tensorflow.lite.examples.poseestimation.SignupRequest
import org.tensorflow.lite.examples.poseestimation.org.tensorflow.lite.examples.poseestimation.SignupResponse
import retrofit2.Response
import retrofit2.http.Body
import retrofit2.http.POST

interface AuthService {
    @POST("login")
    suspend fun login(@Body request: LoginRequest): Response<LoginResponse>

    @POST("signup")
    suspend fun signup(@Body request: SignupRequest): Response<SignupResponse>
}
