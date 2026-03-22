plugins {
    `java`
    application
}

group = "red.vortx"
version = "1.0-SNAPSHOT"

application {
    mainClass.set("red.vortx.neural.Main")
}

repositories {
    mavenCentral()
}

dependencies {
    testImplementation(platform("org.junit:junit-bom:6.0.3"))

    testImplementation("org.junit.jupiter:junit-jupiter-api")
    testRuntimeOnly("org.junit.jupiter:junit-jupiter-engine")

    testRuntimeOnly("org.junit.platform:junit-platform-launcher")

    compileOnly("org.jspecify:jspecify:1.0.0")
}

tasks.getByName<Test>("test") {
    useJUnitPlatform()
}
